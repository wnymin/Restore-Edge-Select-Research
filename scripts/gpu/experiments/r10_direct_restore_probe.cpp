#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "r10_vk_trace_abi.h"

namespace {
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<uint8_t>;
using Tokens = std::vector<llama_token>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
constexpr int positions = 8;
constexpr size_t alignment = 4096;
const char *volatile phase = "start";
int phase_rep = -2, phase_depth = 0, phase_arm = -1;

// Give native faults a local diagnostic without saving address-space contents.
LONG WINAPI fault(EXCEPTION_POINTERS *ex) {
    const auto *r = ex->ExceptionRecord;
    MEMORY_BASIC_INFORMATION info{};
    char image[MAX_PATH]{};
    if (VirtualQuery(r->ExceptionAddress, &info, sizeof(info)) && info.AllocationBase) {
        GetModuleFileNameA(static_cast<HMODULE>(info.AllocationBase), image, DWORD(sizeof(image)));
    }
    const auto offset = reinterpret_cast<uintptr_t>(r->ExceptionAddress) - reinterpret_cast<uintptr_t>(info.AllocationBase);
    std::fprintf(stderr, "r10_native_fault,0x%08lx,%s+0x%llx,rep=%d,depth=%d,arm=%d,phase=%s\n",
                 r->ExceptionCode, image, static_cast<unsigned long long>(offset), phase_rep, phase_depth, phase_arm, phase);
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}

void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
double ms(Clock::time_point epoch) { return std::chrono::duration<double, std::milli>(Clock::now() - epoch).count(); }
size_t padded(size_t n) { return (n + alignment - 1) / alignment * alignment; }

void decode(llama_context *ctx, llama_token *tokens, int n) {
    check(llama_decode(ctx, llama_batch_get_one(tokens, n)) == 0, "decode failed");
}

void reset(llama_context *ctx, bool clear_payload = false) {
    llama_synchronize(ctx);
    llama_memory_clear(llama_get_memory(ctx), clear_payload);
}

void resources(const char *phase, ggml_backend_dev_t device, int depth = 0) {
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(device, &free, &total);
    std::cerr << "r10_direct_resources," << phase << ',' << depth << ',' << free << ',' << total << '\n';
}

Bytes save(llama_context *ctx, llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    Bytes bytes(llama_state_seq_get_size_ext(ctx, 0, flags));
    check(!bytes.empty() && llama_state_seq_get_data_ext(ctx, bytes.data(), bytes.size(), 0, flags) == bytes.size(),
          "could not serialize sequence state");
    return bytes;
}

std::vector<float> logits(llama_context *ctx, llama_token *continuation, int vocab) {
    std::vector<float> output;
    output.reserve(size_t(positions) * vocab);
    for (int i = 0; i < positions; ++i) {
        decode(ctx, continuation + i, 1);
        const float *last = llama_get_logits_ith(ctx, -1);
        check(last && std::all_of(last, last + vocab, [](float x) { return std::isfinite(x); }), "invalid logits");
        output.insert(output.end(), last, last + vocab);
    }
    return output;
}

struct Compare { int top = 0; double max_error = 0.0; };
Compare compare(const std::vector<float> &a, const std::vector<float> &b, int vocab) {
    Compare result;
    for (int i = 0; i < positions; ++i) {
        const float *x = a.data() + i * vocab, *y = b.data() + i * vocab;
        result.top += std::max_element(x, x + vocab) - x == std::max_element(y, y + vocab) - y;
        for (int j = 0; j < vocab; ++j) result.max_error = std::max(result.max_error, std::abs(double(x[j]) - y[j]));
    }
    return result;
}

struct Scratch {
    std::filesystem::path path;
    ~Scratch() { std::error_code ignored; std::filesystem::remove(path, ignored); }
    void stage(const uint8_t *payload, size_t n) const {
        const Bytes pad(padded(n) - n);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(payload), n);
        if (!pad.empty()) out.write(reinterpret_cast<const char *>(pad.data()), pad.size());
        out.close();
        check(bool(out), "could not stage padded SSD snapshot");
    }
    size_t read(void *dest, size_t n) const {
        HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
        check(f != INVALID_HANDLE_VALUE, "could not open unbuffered SSD snapshot");
        IO_COUNTERS before{}, after{};
        const BOOL measured_before = GetProcessIoCounters(GetCurrentProcess(), &before);
        DWORD got = 0;
        const BOOL ok = ReadFile(f, dest, DWORD(n), &got, nullptr);
        const BOOL measured_after = GetProcessIoCounters(GetCurrentProcess(), &after);
        CloseHandle(f);
        check(ok && got == n, "unbuffered SSD read failed or returned short");
        check(measured_before && measured_after && after.ReadTransferCount >= before.ReadTransferCount &&
              after.ReadTransferCount - before.ReadTransferCount >= got, "SSD read is missing from process I/O accounting");
        return size_t(after.ReadTransferCount - before.ReadTransferCount);
    }
};

struct Aligned {
    explicit Aligned(size_t bytes) : ptr(VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)) {
        check(ptr, "aligned host allocation failed");
    }
    ~Aligned() { VirtualFree(ptr, 0, MEM_RELEASE); }
    void *ptr;
};

template<typename F> F proc(ggml_backend_reg_t reg, const char *name) {
    void *p = ggml_backend_reg_get_proc_address(reg, name);
    F fn = nullptr;
    static_assert(sizeof(p) == sizeof(fn));
    std::memcpy(&fn, &p, sizeof(fn));
    check(fn, name);
    return fn;
}

struct Observer {
    explicit Observer(const std::filesystem::path &path) : dll(LoadLibraryW(std::filesystem::absolute(path).c_str())) {
        check(dll, "could not load the explicit Vulkan observer DLL");
        const auto a = GetProcAddress(dll, "r10_trace_arm"), w = GetProcAddress(dll, "r10_trace_wait");
        static_assert(sizeof(a) == sizeof(arm) && sizeof(w) == sizeof(wait));
        std::memcpy(&arm, &a, sizeof(a));
        std::memcpy(&wait, &w, sizeof(w));
        check(arm && wait, "missing explicit observer functions");
    }
    ~Observer() { FreeLibrary(dll); }
    HMODULE dll;
    R10TraceArm arm = nullptr;
    R10TraceWait wait = nullptr;
};
}

int main(int argc, char **argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    SetUnhandledExceptionFilter(fault);
    check(argc == 8, "usage: r10_direct_restore_probe MODEL DLLS DEVICE NEW_SCRATCH REPS MAX_DEPTH OBSERVER_DLL");
    check(std::getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") && std::getenv("GGML_VK_R10_CONCURRENT_BUFFERS"),
          "set GGML_VK_ASYNC_USE_TRANSFER_QUEUE and GGML_VK_R10_CONCURRENT_BUFFERS for this process");
    const int reps = std::stoi(argv[5]), max_depth = std::stoi(argv[6]);
    check(reps > 0 && (max_depth == 512 || max_depth == 1024 || max_depth == 1536) &&
          !std::filesystem::exists(argv[4]), "expected a fresh scratch path, positive repeats, and supported depth");
    Scratch disk{std::filesystem::absolute(argv[4])};
    const Observer observer(argv[7]);
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name(argv[3]), nullptr};
    check(devices[0], "requested Vulkan device missing");
    resources("before-model", devices[0]);
    const auto host_type = ggml_backend_dev_host_buffer_type(devices[0]);
    const auto reg = ggml_backend_dev_backend_reg(devices[0]);
    check(host_type && reg, "pinned host buffers or Vulkan registry unavailable");
    const auto copy = proc<bool (*)(ggml_backend_t, ggml_tensor *, const void *, size_t, size_t)>(reg, "ggml_backend_vk_r10_copy");
    const auto submit = proc<uint64_t (*)(ggml_backend_t)>(reg, "ggml_backend_vk_r10_submit");
    const auto wait = proc<int (*)(ggml_backend_t, uint64_t, uint64_t)>(reg, "ggml_backend_vk_r10_wait");
    const auto bind = proc<bool (*)(ggml_backend_t, uint64_t)>(reg, "ggml_backend_vk_r10_bind");
    const std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> upload(ggml_backend_dev_init(devices[0], nullptr), ggml_backend_free);
    check(bool(upload), "dedicated upload backend missing");
    const std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> marker(ggml_backend_dev_init(devices[0], nullptr), ggml_backend_free);
    check(bool(marker), "compute marker backend missing");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    check(bool(model), "could not load model");
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    const Context consumer(llama_init_from_model(model.get(), cp), llama_free);
    const Context predecessor(llama_init_from_model(model.get(), cp), llama_free);
    check(consumer && predecessor, "could not initialize separate predecessor and successor contexts");
    resources("contexts-ready", devices[0]);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::mt19937 random(14087), order(142457);
    uint64_t capture_id = 0;
    std::cout << "rep,depth,order,arm,checkpoint_bytes,attention_encoded_bytes,ssd_bytes,process_read_bytes,span_count,dma_bytes,"
                 "pred_submitted_ms,pred_done_bound_ms,pred_fence_pending_at_dma_ready,read_done_ms,host_ready_ms,dma_submit_ms,dma_ready_ms,"
                 "bind_ms,consume_ms,consumed_ms,top_matches,max_error,after_consume_bytes_equal,timeline_value\n"
              << std::setprecision(10);

    for (int rep = -1; rep < reps; ++rep) {
        Tokens prompt(max_depth + positions), other(512);
        for (auto &t : prompt) t = 1000 + random() % 14000;
        for (auto &t : other) t = 1000 + random() % 14000;
        for (int depth : {512, 1024, 1536}) {
            if (depth > max_depth) break;
            phase_rep = rep;
            phase_depth = depth;
            phase_arm = -1;
            auto *dest = consumer.get();
            phase = "reference-prefill";
            reset(dest);
            for (int i = 0; i < depth; i += 256) decode(dest, prompt.data() + i, 256);
            phase = "reference-snapshot";
            const Bytes image = save(dest), recurrent = save(dest, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            constexpr size_t header = sizeof(uint32_t) + sizeof(llama_seq_id);
            check(recurrent.size() > header && recurrent.size() < image.size() &&
                  std::equal(recurrent.begin(), recurrent.begin() + header, image.begin()) &&
                  std::equal(recurrent.rbegin(), recurrent.rend() - header, image.rbegin()), "unexpected hybrid-state layout");
            const size_t ram_bytes = image.size() - recurrent.size() + header;
            const size_t read_bytes = padded(image.size() - ram_bytes);
            disk.stage(image.data() + ram_bytes, image.size() - ram_bytes);
            const Aligned bounce(read_bytes);
            phase = "reference-continuation";
            const auto expected = logits(dest, prompt.data() + depth, vocab);
            const Bytes expected_state = save(dest);
            phase = "allocating-pinned";
            const Buffer pinned(ggml_backend_buft_alloc_buffer(host_type, padded(image.size())), ggml_backend_buffer_free);
            check(pinned && ggml_backend_buffer_get_type(pinned.get()) == host_type, "pinned allocation failed or fell back");
            if (rep == 0) resources("pinned-and-contexts", devices[0], depth);
            auto *const staging = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pinned.get()));
            check(staging, "pinned host mapping missing");
            std::array<std::string, 3> arms{"ram-isolated", "ram-overlap", "split-ssd-overlap"};
            std::shuffle(arms.begin(), arms.end(), order);
            for (size_t index = 0; index < arms.size(); ++index) {
                phase_arm = int(index);
                const auto &arm = arms[index];
                const bool overlap = arm != "ram-isolated", ssd = arm == "split-ssd-overlap";
                phase = "clearing-old-state";
                reset(dest, true);
                reset(predecessor.get());
                Bytes source(image.size());
                const std::unique_ptr<ggml_backend_event, decltype(&ggml_backend_event_free)> previous(
                    overlap ? ggml_backend_event_new(devices[0]) : nullptr, ggml_backend_event_free);
                check(!overlap || bool(previous), "could not allocate the Vulkan compute queue marker");
                const auto epoch = Clock::now();
                double pred_submit = -1, pred_done = -1, read_done = -1;
                int pred_pending = -1;
                size_t process_read = 0;
                if (overlap) {
                    phase = "predecessor-submit";
                    decode(predecessor.get(), other.data(), int(other.size()));
                    check(observer.arm(++capture_id) > 0, "Vulkan observer not active for compute marker");
                    ggml_backend_event_record(previous.get(), marker.get());
                    observer.arm(0);
                    pred_submit = ms(epoch);
                }
                phase = "restore-prepare";
                if (ssd) {
                    process_read = disk.read(bounce.ptr, read_bytes);
                    read_done = ms(epoch);
                    std::memcpy(source.data(), image.data(), ram_bytes);
                    std::memcpy(source.data() + ram_bytes, bounce.ptr, image.size() - ram_bytes);
                } else {
                    std::memcpy(source.data(), image.data(), image.size());
                }
                std::array<llama_r10_restore_span, 512> spans{};
                size_t count = 0, bytes = 0;
                check(llama_r10_prepare_seq_restore(dest, source.data(), source.size(), 0, spans.data(), spans.size(), &count) == source.size() &&
                      count && llama_memory_seq_pos_max(llama_get_memory(dest), 0) == depth - 1, "restore span planning failed");
                // Pack only payloads in pinned memory; metadata lengths need not be 4-byte aligned.
                for (size_t i = 0; i < count; ++i) {
                    const auto &sp = spans[i];
                    check(sp.src >= source.data() && sp.src <= source.data() + source.size() &&
                          sp.size <= size_t(source.data() + source.size() - sp.src) &&
                          sp.size % 4 == 0 && bytes <= source.size() && sp.size <= source.size() - bytes,
                          "prepared payload bounds or alignment are invalid");
                    std::memcpy(staging + bytes, sp.src, sp.size);
                    bytes += sp.size;
                }
                const auto host_ready = ms(epoch);
                phase = "transfer-submit";
                size_t pos = 0;
                for (size_t i = 0; i < count; ++i) {
                    const auto &sp = spans[i];
                    check(copy(upload.get(), sp.tensor, staging + pos, sp.offset, sp.size), "direct upload span rejected");
                    pos += sp.size;
                }
                const uint64_t value = submit(upload.get());
                check(value, "missing transfer timeline signal");
                const auto dma_submit = ms(epoch);
                phase = "transfer-wait";
                check(wait(upload.get(), value, 5'000'000'000ULL) == 1, "waiting for transfer failed or timed out");
                const auto dma_ready = ms(epoch);
                if (previous) {
                    phase = "predecessor-fence";
                    R10TraceRecord signal{};
                    const int status = observer.wait(capture_id, 0, &signal);
                    check(signal.queue && signal.timeline_value && (status == R10_TRACE_WAIT_TIMEOUT || status == R10_TRACE_OK),
                          "could not poll predecessor's recorded device wait point");
                    pred_pending = status == R10_TRACE_WAIT_TIMEOUT;
                    check(observer.wait(capture_id, 5000, &signal) == R10_TRACE_OK, "predecessor device marker failed to signal");
                    pred_done = ms(epoch);
                    ggml_backend_synchronize(marker.get());
                    llama_synchronize(predecessor.get());
                }
                phase = "successor-bind";
                check(bind(upload.get(), value), "could not bind signal to the consuming queue");
                const auto bound = ms(epoch);
                // There is no state readback between upload and the first actual consumer.
                const auto consume = ms(epoch);
                phase = "successor-consume";
                const auto actual = logits(dest, prompt.data() + depth, vocab);
                const auto done = ms(epoch);
                const auto diff = compare(expected, actual, vocab);
                phase = "successor-readback";
                const bool state = save(dest) == expected_state;
                std::cout << rep << ',' << depth << ',' << index << ',' << arm << ',' << image.size() << ',' << ram_bytes << ','
                          << (ssd ? read_bytes : 0) << ',' << process_read << ',' << count << ',' << bytes << ',' << pred_submit << ',' << pred_done << ','
                          << pred_pending << ',' << read_done << ',' << host_ready << ',' << dma_submit << ',' << dma_ready << ',' << bound << ',' << consume << ','
                          << done << ',' << diff.top << ',' << diff.max_error << ',' << state << ',' << value << std::endl;
                check(diff.top == positions && diff.max_error == 0 && state, "consumer output or final state differs from recompute");
            }
            phase = "release-checkpoint-buffers";
        }
    }
    resources("end", devices[0]);
    phase = "shutdown";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
