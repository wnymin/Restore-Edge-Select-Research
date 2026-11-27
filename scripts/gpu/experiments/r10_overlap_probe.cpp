#include "llama.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include "r10_vk_trace_abi.h"

namespace {
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<uint8_t>;
constexpr size_t alignment = 4096;

void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int64_t counter(bool frequency = false) {
    LARGE_INTEGER n{};
    check(frequency ? QueryPerformanceFrequency(&n) : QueryPerformanceCounter(&n), "performance counter failed");
    return n.QuadPart;
}

struct Scratch {
    std::filesystem::path path;
    ~Scratch() { std::error_code ignored; std::filesystem::remove(path, ignored); }
};

struct Aligned {
    explicit Aligned(size_t len) : ptr(VirtualAlloc(nullptr, len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)) {
        check(ptr, "VirtualAlloc failed");
    }
    ~Aligned() { VirtualFree(ptr, 0, MEM_RELEASE); }
    void *ptr;
};

void write_scratch(const std::filesystem::path &path, const Bytes &bytes, size_t padded) {
    const Bytes pad(padded - bytes.size());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    file.write(reinterpret_cast<const char *>(pad.data()), pad.size());
    file.close();
    check(bool(file), "failed writing padded file");
}

DWORD read_exact(HANDLE file, void *buf, DWORD size) {
    DWORD got = 0;
    check(SetFilePointerEx(file, {}, nullptr, FILE_BEGIN), "file rewind failed");
    check(ReadFile(file, buf, size, &got, nullptr) && got == size, "unbuffered read failed or short");
    return got;
}

void resources(const char *phase, ggml_backend_dev_t device) {
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(device, &free, &total);
    PROCESS_MEMORY_COUNTERS_EX mem{};
    check(K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&mem),
                                  sizeof(mem)), "process memory snapshot failed");
    std::cerr << "r10_resources," << phase << ',' << free << ',' << total << ','
              << mem.WorkingSetSize << ',' << mem.PeakWorkingSetSize << ',' << mem.PrivateUsage << '\n';
}

void decode(llama_context *ctx, llama_token *tokens, int count) {
    check(llama_decode(ctx, llama_batch_get_one(tokens, count)) == 0, "model decode failed");
}

struct ComputeSample {
    double start = -1, submitted = -1, done = -1;
    int top = -1;
};

struct TransferSample {
    double start = -1, read_done = -1, host_ready = -1, queued = -1, done = -1;
    DWORD actual_read = 0;
};
}

int main(int argc, char **argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    check(argc == 7 || argc == 8, "usage: r10_overlap_probe MODEL BACKENDS DEVICE NEW_SCRATCH REPEATS SEED [VULKAN_LAYER_DLL]");
    const bool traced = argc == 8;
    HMODULE module = traced ? LoadLibraryW(std::filesystem::absolute(argv[7]).c_str()) : nullptr;
    check(!traced || module, "could not load explicit Vulkan observer library");
    R10TraceArm trace_arm = nullptr;
    R10TraceWait trace_wait = nullptr;
    if (traced) {
        const auto arm_proc = GetProcAddress(module, "r10_trace_arm"), wait_proc = GetProcAddress(module, "r10_trace_wait");
        static_assert(sizeof(arm_proc) == sizeof(trace_arm) && sizeof(wait_proc) == sizeof(trace_wait));
        std::memcpy(&trace_arm, &arm_proc, sizeof(trace_arm));
        std::memcpy(&trace_wait, &wait_proc, sizeof(trace_wait));
        check(trace_arm && trace_wait, "explicit Vulkan observer API missing");
    }
    const int reps = std::stoi(argv[5]);
    check(reps > 0 && !std::filesystem::exists(argv[4]), "positive repeats and a new scratch path required");
    Scratch scratch{argv[4]};
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name(argv[3]), nullptr};
    check(devices[0], "requested GPU backend missing");
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(devices[0], &props);
    check(props.caps.async && props.caps.host_buffer, "async and host-buffer capabilities required");
    const auto host_type = ggml_backend_dev_host_buffer_type(devices[0]);
    check(host_type, "missing pinned host buffer type");
    std::cerr << "r10_resources,phase,gpu_free_bytes,gpu_total_bytes,rss_bytes,peak_rss_bytes,private_bytes\n";
    resources("before-model", devices[0]);
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    check(bool(model), "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = cp.n_ubatch = 512;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    check(bool(ctx), "model context init failed");
    const auto memory = llama_get_memory(ctx.get());
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::mt19937 random(std::stoul(argv[6]));
    std::vector<llama_token> prompt(512);
    for (auto &token : prompt) token = 1000 + random() % 14000;
    decode(ctx.get(), prompt.data(), prompt.size());
    llama_synchronize(ctx.get());
    // Transfer real serialized recurrent/conv state into a separate device buffer.
    // This tests the transport against model compute. It does not attach the buffer to the model.
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t bytes = llama_state_seq_get_size_ext(ctx.get(), 0, flags);
    Bytes payload(bytes);
    check(bytes > 0 && llama_state_seq_get_data_ext(ctx.get(), payload.data(), bytes, 0, flags) == bytes,
          "partial-state capture failed");
    const size_t padded = (bytes + alignment - 1) / alignment * alignment;
    check(padded < MAXDWORD, "payload exceeds Win32 read size");
    write_scratch(scratch.path, payload, padded);
    HANDLE input = CreateFileW(scratch.path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING, nullptr);
    check(input != INVALID_HANDLE_VALUE, "failed opening unbuffered checkpoint");
    struct Close { HANDLE file; ~Close() { CloseHandle(file); } } close{input};
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    Buffer ram(ggml_backend_buft_alloc_buffer(host_type, padded), ggml_backend_buffer_free);
    Buffer disk(ggml_backend_buft_alloc_buffer(host_type, padded), ggml_backend_buffer_free);
    check(ram && disk && ggml_backend_buffer_get_type(ram.get()) == host_type &&
          ggml_backend_buffer_get_type(disk.get()) == host_type, "pinned host buffers unavailable");
    void *ram_data = ggml_backend_buffer_get_base(ram.get());
    void *disk_data = ggml_backend_buffer_get_base(disk.get());
    check(ram_data && disk_data, "host buffers have no address");
    std::memcpy(ram_data, payload.data(), bytes);
    const bool disk_direct = reinterpret_cast<uintptr_t>(disk_data) % alignment == 0;
    auto bounce = disk_direct ? nullptr : std::make_unique<Aligned>(padded);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>
        transport(ggml_backend_dev_init(devices[0], nullptr), ggml_backend_free);
    check(bool(transport), "transfer backend init failed");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> metadata(
        ggml_init({ggml_tensor_overhead(), nullptr, true}), ggml_free);
    check(bool(metadata), "tensor metadata init failed");
    auto *target = ggml_new_tensor_1d(metadata.get(), GGML_TYPE_I8, bytes);
    Buffer device(ggml_backend_alloc_ctx_tensors(metadata.get(), transport.get()), ggml_backend_buffer_free);
    check(device && !ggml_backend_buffer_is_host(device.get()), "device tensor allocation failed");
    resources("after-warmup-and-allocation", devices[0]);

    std::cout << "rep,suffix,order,arm,payload_bytes,device_bytes,disk_direct,read_bytes,compute_start_ms,"
                 "compute_submit_ms,compute_done_ms,top,transfer_start_ms,read_done_ms,host_ready_ms,"
                 "transfer_queued_ms,transfer_done_ms,device_equal,trace_status,signal_submitted_ms,"
                 "signal_observed_ms,timeline_value\n" << std::setprecision(10);
    const int64_t qpc_frequency = counter(true);
    uint64_t capture_id = 0;
    for (int rep = -1; rep < reps; ++rep) {
        for (auto &token : prompt) token = 1000 + random() % 14000;
        for (int suffix : {128, 512}) {
            std::array<std::string, 7> arms = {"compute-a", "compute-a-repeat", "ram-only", "ssd-only",
                                               "joint-ram", "joint-ram-after-submit", "joint-ssd"};
            std::shuffle(arms.begin(), arms.end(), random);
            int top_control = -1;
            for (size_t i = 0; i < arms.size(); ++i) {
                const auto &arm = arms[i];
                const bool compute = arm != "ram-only" && arm != "ssd-only";
                const bool transfer = arm != "compute-a" && arm != "compute-a-repeat";
                const bool ssd = arm == "ssd-only" || arm == "joint-ssd";
                const bool after_submit = arm == "joint-ram-after-submit";
                if (compute) llama_memory_clear(memory, false);
                Clock::time_point epoch;
                const uint64_t current_id = ++capture_id;
                int64_t epoch_qpc = 0;
                R10TraceRecord signal{};
                std::future<int> observation;
                std::promise<void> ready, start, submitted;
                auto start_signal = start.get_future();
                auto submit_signal = submitted.get_future();
                std::future<TransferSample> job;
                struct Signal {
                    std::promise<void> &promise;
                    bool sent = false;
                    void release() { if (!sent) { promise.set_value(); sent = true; } }
                    ~Signal() { release(); }
                } submitted_signal{submitted};
                if (transfer) {
                    if (traced) observation = std::async(std::launch::async, [&] { return trace_wait(current_id, 3000, &signal); });
                    job = std::async(std::launch::async, [&]() {
                        TransferSample t;
                        ready.set_value();
                        start_signal.wait();
                        if (after_submit) submit_signal.wait();
                        t.start = ms(epoch);
                        if (ssd) {
                            void *read_buffer = disk_direct ? disk_data : bounce->ptr;
                            t.actual_read = read_exact(input, read_buffer, DWORD(padded));
                            t.read_done = ms(epoch);
                            if (!disk_direct) std::memcpy(disk_data, read_buffer, bytes);
                        }
                        t.host_ready = ms(epoch);
                        ggml_backend_tensor_set_async(transport.get(), target, ssd ? disk_data : ram_data, 0, bytes);
                        t.queued = ms(epoch);
                        if (traced) check(trace_arm(current_id) > 0, "loader did not activate the explicit observer");
                        ggml_backend_synchronize(transport.get());
                        t.done = ms(epoch);
                        if (traced) trace_arm(0);
                        return t;
                    });
                    ready.get_future().wait();
                }
                epoch_qpc = counter();
                epoch = Clock::now();
                start.set_value();
                ComputeSample c;
                if (compute) {
                    c.start = ms(epoch);
                    decode(ctx.get(), prompt.data(), suffix);
                    c.submitted = ms(epoch);
                    submitted_signal.release();
                    llama_synchronize(ctx.get());
                    c.done = ms(epoch);
                    const float *logits = llama_get_logits_ith(ctx.get(), -1);
                    check(logits && std::all_of(logits, logits + vocab, [](float x) { return std::isfinite(x); }),
                          "compute has no valid logits");
                    c.top = int(std::max_element(logits, logits + vocab) - logits);
                    check(top_control == -1 || top_control == c.top, "greedy token differs across paired arms");
                    top_control = c.top;
                }
                TransferSample t;
                int device_equal = -1;
                int trace_status = -1;
                double signal_submitted = -1, signal_observed = -1;
                if (transfer) {
                    t = job.get();
                    if (traced) {
                        trace_status = observation.get();
                        if (trace_status == R10_TRACE_OK) {
                            check(signal.queue && signal.timeline_value && signal.observed_ready_qpc >= signal.submitted_qpc,
                                  "invalid timeline completion record");
                            signal_submitted = 1000.0 * (signal.submitted_qpc - epoch_qpc) / qpc_frequency;
                            signal_observed = 1000.0 * (signal.observed_ready_qpc - epoch_qpc) / qpc_frequency;
                        }
                    }
                    // Clear after each measured/readback arm, so stale device content cannot pass the next byte check.
                    Bytes actual(bytes);
                    ggml_backend_tensor_get(target, actual.data(), 0, bytes);
                    device_equal = actual == payload;
                    ggml_backend_tensor_memset(target, 0, 0, bytes);
                    check(device_equal, "device readback differs from saved payload");
                }
                std::cout << rep << ',' << suffix << ',' << i << ',' << arm << ',' << bytes << ','
                          << ggml_backend_buffer_get_size(device.get()) << ',' << disk_direct << ',' << t.actual_read << ','
                          << c.start << ',' << c.submitted << ',' << c.done << ',' << c.top << ',' << t.start << ','
                          << t.read_done << ',' << t.host_ready << ',' << t.queued << ',' << t.done << ','
                          << device_equal << ',' << trace_status << ',' << signal_submitted << ',' << signal_observed
                          << ',' << signal.timeline_value << std::endl;
            }
        }
    }
    resources("after-all-arms", devices[0]);
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
