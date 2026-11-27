#include "r10_catalog_source.h"
#include "r10_catalog_target.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>

namespace {
using namespace r10cat;
using Clock = std::chrono::steady_clock;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
constexpr size_t checkpoint_bytes = 52690944;
constexpr size_t compute_chunk = 256;
constexpr size_t probe_steps = 8;
const char *volatile phase = "startup";

LONG WINAPI fault(EXCEPTION_POINTERS *ex) {
    std::fprintf(stderr, "r10_catalog_fault,0x%08lx,phase=%s\n", ex->ExceptionRecord->ExceptionCode, phase);
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}

void reset(llama_context *ctx) {
    llama_synchronize(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
}

void decode(llama_context *ctx, Tokens &input, size_t start, size_t n) {
    need(n && start <= input.size() && n <= input.size() - start, "invalid catalogue decode range");
    need(llama_decode(ctx, llama_batch_get_one(input.data() + start, int(n))) == 0, "catalogue decode failed");
}

Bytes state(llama_context *ctx, llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    Bytes bytes(llama_state_seq_get_size_ext(ctx, 0, flags));
    need(!bytes.empty() && llama_state_seq_get_data_ext(ctx, bytes.data(), bytes.size(), 0, flags) == bytes.size(),
         "could not read catalogue checkpoint state");
    return bytes;
}

std::vector<float> continuation(llama_context *ctx, Tokens input, size_t vocabulary) {
    std::vector<float> result;
    result.reserve(input.size() * vocabulary);
    for (size_t i = 0; i < input.size(); ++i) {
        decode(ctx, input, i, 1);
        const auto *logits = llama_get_logits_ith(ctx, -1);
        need(logits && std::all_of(logits, logits + vocabulary, [](float v) { return std::isfinite(v); }),
             "nonfinite catalogue continuation logits");
        result.insert(result.end(), logits, logits + vocabulary);
    }
    return result;
}

struct Packed { Meta meta; std::map<size_t, Bytes> objects; };

Packed pack(llama_context *ctx, const Plan &plan, const Candidate &c, const Bytes &full, size_t recurrent_start) {
    Shape spans{};
    const auto n = prepare_spans(ctx, full, c.depth, spans);
    Packed result{{c.depth, full, {}}, {}};
    for (auto id : c.deps) result.objects[id].reserve(plan.objects.at(id).bytes);
    const size_t blocks = c.depth / plan.block;
    for (size_t i = 0; i < n; ++i) {
        const auto &s = spans[i];
        need(s.src != nullptr && reinterpret_cast<uintptr_t>(s.src) >= reinterpret_cast<uintptr_t>(full.data()),
             "restore source before state bytes");
        const auto at = reinterpret_cast<uintptr_t>(s.src) - reinterpret_cast<uintptr_t>(full.data());
        need(s.tensor && s.tensor->buffer && s.size && s.offset % 4 == 0 && s.size % 4 == 0 &&
             at <= full.size() && s.size <= full.size() - at && (at >= recurrent_start || s.size <= recurrent_start - at),
             "invalid tensor or crossed state encoding boundary");
        const bool kv = at < recurrent_start;
        const size_t count = kv ? blocks : 1;
        need(s.size % count == 0 && (s.size / count) % 4 == 0, "native tensor is not divisible into aligned KV blocks");
        Region r{at, s.size, ggml_get_name(s.tensor), {}};
        const size_t part_bytes = s.size / count;
        for (size_t j = 0; j < count; ++j) {
            const auto id = kv ? c.deps.at(j) : c.deps.back();
            auto &blob = result.objects.at(id);
            const auto *begin = full.data() + at + j * part_bytes;
            r.parts.push_back({id, blob.size(), part_bytes});
            blob.insert(blob.end(), begin, begin + part_bytes);
        }
        std::fill_n(result.meta.skeleton.begin() + at, s.size, uint8_t(0));
        result.meta.regions.push_back(std::move(r));
    }
    for (auto id : c.deps) need(result.objects.at(id).size() == plan.objects.at(id).bytes, "native object size differs from plan");
    return result;
}

void write_object(const fs::path &path, const Bytes &bytes) {
    need(!fs::exists(path), "refusing to overwrite a catalogue object");
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
    f.close();
    need(bool(f), "could not write catalogue object bytes");
}

void read_ram_source(const fs::path &path, uint8_t *target, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    need(bool(f.read(reinterpret_cast<char *>(target), std::streamsize(bytes))) && f.peek() == std::char_traits<char>::eof(),
         "catalogue RAM source artifact has the wrong length");
}

template<typename F> F symbol(ggml_backend_reg_t registry, const char *name) {
    auto ptr = ggml_backend_reg_get_proc_address(registry, name);
    F fn{};
    static_assert(sizeof(fn) == sizeof(ptr));
    std::memcpy(&fn, &ptr, sizeof(fn));
    need(fn != nullptr, name);
    return fn;
}

class Uploader {
public:
    explicit Uploader(ggml_backend_dev_t dev) :
        backend_(ggml_backend_dev_init(dev, nullptr), ggml_backend_free),
        copy_(symbol<decltype(copy_)>(ggml_backend_dev_backend_reg(dev), "ggml_backend_vk_r10_copy")),
        submit_(symbol<decltype(submit_)>(ggml_backend_dev_backend_reg(dev), "ggml_backend_vk_r10_submit")),
        wait_(symbol<decltype(wait_)>(ggml_backend_dev_backend_reg(dev), "ggml_backend_vk_r10_wait")),
        bind_(symbol<decltype(bind_)>(ggml_backend_dev_backend_reg(dev), "ggml_backend_vk_r10_bind")),
        host_(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(dev), checkpoint_bytes), ggml_backend_buffer_free),
        disk_(checkpoint_bytes), device_type_(ggml_backend_dev_buffer_type(dev)) {
        need(backend_ && host_ && ggml_backend_buffer_get_type(host_.get()) == ggml_backend_dev_host_buffer_type(dev),
             "could not allocate native upload resources");
        pinned_ = static_cast<uint8_t *>(ggml_backend_buffer_get_base(host_.get()));
        need(pinned_, "native upload buffer has no host address");
    }

    void restore(llama_context *ctx, const fs::path &root, const Plan &plan, const Candidate &c, const Meta &meta) {
        const auto target = bind_target(ctx, meta, device_type_);
        for (auto id : c.deps) {
            const auto &o = plan.objects.at(id);
            const auto file = path_for(root, id, false);
            need(o.bytes <= checkpoint_bytes, "catalogue upload exceeds pinned allocation");
            if (o.kind == 'R') {
                disk_.start(file, o.bytes);
                const auto result = disk_.poll(true); // catalogue qualification may wait; live FIFO will poll
                need(result && result->bytes == o.bytes, "incomplete qualification SSD read");
                std::memcpy(pinned_, disk_.payload(o.bytes), o.bytes);
            } else {
                read_ram_source(file, pinned_, o.bytes); // preparation only; runtime will use resident RAM
            }
            size_t sent = 0;
            for (const auto &part : target.at(id)) {
                const auto &r = part.where;
                need(r.object_offset == sent && copy_(backend_.get(), part.tensor, pinned_ + r.object_offset,
                     part.tensor_offset, r.bytes), "catalogue object upload rejected");
                sent += r.bytes;
            }
            need(sent == o.bytes, "native object upload differs from sealed bytes");
            const auto ticket = submit_(backend_.get());
            need(ticket && wait_(backend_.get(), ticket, 5'000'000'000ULL) == 1 && bind_(backend_.get(), ticket),
                 "native catalogue upload did not become visible");
        }
    }
private:
    Backend backend_;
    bool (*copy_)(ggml_backend_t, ggml_tensor *, const void *, size_t, size_t);
    uint64_t (*submit_)(ggml_backend_t);
    int (*wait_)(ggml_backend_t, uint64_t, uint64_t);
    bool (*bind_)(ggml_backend_t, uint64_t);
    Buffer host_;
    DiskRead disk_;
    ggml_backend_buffer_type_t device_type_;
    uint8_t *pinned_{};
};

void build(llama_context *ctx, Uploader &uploader, const Plan &plan, const fs::path &root, size_t vocabulary) {
    need(fs::create_directory(root), "catalogue output must be a new directory with an existing parent");
    std::set<size_t> written;
    std::cout << "checkpoint,depth,dependencies,full_bytes,metadata_bytes,shared_identity_checks,preparation_us,exact_state,exact_logits_positions,exact_final_state\n";
    for (const auto &[id, owner] : plan.checkpoint_owner) {
        const auto start = Clock::now();
        const auto depth = plan.checkpoint_depth.at(id);
        const auto &row = plan.requests.at(owner);
        const auto &c = row.options.at(depth);
        need(c.deps.back() == id, "catalogue owner changed before preparation");
        phase = "independent reference";
        reset(ctx);
        auto tokens = row.tokens;
        need(std::all_of(tokens.begin(), tokens.end(), [vocabulary](llama_token t) { return size_t(t) < vocabulary; }),
             "catalogue token exceeds model vocabulary");
        for (size_t i = 0; i < depth; i += compute_chunk) decode(ctx, tokens, i, compute_chunk);
        const auto full = state(ctx);
        size_t first_recurrent = 0;
        {
            const auto rec = state(ctx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            constexpr size_t prefix = sizeof(uint32_t) + sizeof(llama_seq_id);
            need(rec.size() > prefix && full.size() > rec.size() && std::equal(rec.begin(), rec.begin() + prefix, full.begin()) &&
                 std::equal(rec.rbegin(), rec.rend() - prefix, full.rbegin()), "changed native checkpoint encoding");
            first_recurrent = full.size() - rec.size() + prefix;
        }
        const Tokens probe(tokens.begin(), tokens.begin() + probe_steps);
        const auto expected_logits = continuation(ctx, probe, vocabulary);
        const auto expected_final = state(ctx);
        phase = "export objects";
        reset(ctx);
        size_t shared = 0;
        const auto descriptor = path_for(root, id, true);
        {
            const auto packed = pack(ctx, plan, c, full, first_recurrent);
            for (auto object : c.deps) {
                const auto &b = packed.objects.at(object);
                const auto file = path_for(root, object, false);
                if (written.insert(object).second) {
                    write_object(file, b);
                } else {
                    Bytes earlier(b.size());
                    read_ram_source(file, earlier.data(), earlier.size());
                    need(b == earlier, "putative shared KV differs across independently recomputed checkpoints");
                    ++shared;
                }
            }
            store_meta(descriptor, packed.meta);
        }
        phase = "restore and compare";
        reset(ctx);
        {
            const auto disk_meta = load_meta(descriptor, plan, c);
            uploader.restore(ctx, root, plan, c, disk_meta);
        }
        need(state(ctx) == full, "catalogue upload did not reproduce exact bytes");
        const auto actual_logits = continuation(ctx, probe, vocabulary);
        need(actual_logits.size() == expected_logits.size() &&
             std::memcmp(actual_logits.data(), expected_logits.data(), actual_logits.size() * sizeof(float)) == 0,
             "catalogue continuation logits differ from independent recomputation");
        need(state(ctx) == expected_final, "catalogue continuation state differs from independent recomputation");
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
        std::cout << id << ',' << depth << ',' << c.deps.size() << ',' << full.size() << ',' << fs::file_size(descriptor)
                  << ',' << shared << ',' << elapsed << ",1," << probe_steps << ",1\n" << std::flush;
    }
    need(written.size() == plan.objects.size(), "native catalogue is incomplete");
}
}

int main(int argc, const char **argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    SetUnhandledExceptionFilter(fault);
    need(argc == 6, "usage: r10_prepare_catalog MODEL DLLDIR DEVICE PLAN NEW_CATALOGUE_DIR");
    const auto plan = read_plan(argv[4]);
    need(!plan.checkpoint_owner.empty(), "catalogue has no positive checkpoints");
    need(!fs::exists(argv[5]) && fs::is_directory(fs::absolute(argv[5]).parent_path()), "new output with existing parent required");
    need(std::getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") && std::getenv("GGML_VK_R10_CONCURRENT_BUFFERS"),
         "explicit async transfer and concurrent buffers opt-ins required");
    phase = "initialize model";
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devices[]{ggml_backend_dev_by_name(argv[3]), nullptr};
    need(devices[0] && ggml_backend_dev_host_buffer_type(devices[0]), "requested Vulkan device and pinned storage unavailable");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    need(bool(model), "could not load catalogue model");
    auto cp = llama_context_default_params();
    cp.n_ctx = uint32_t(plan.context);
    cp.n_batch = cp.n_ubatch = compute_chunk;
    cp.n_seq_max = 1;
    const Context ctx(llama_init_from_model(model.get(), cp), llama_free);
    need(bool(ctx), "could not create native catalogue model context");
    auto vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    need(vocabulary > 0, "invalid model vocabulary");
    Uploader uploader(devices[0]);
    build(ctx.get(), uploader, plan, fs::absolute(argv[5]), size_t(vocabulary));
    return 0;
} catch (const std::exception &error) {
    std::cerr << "r10 catalogue at " << phase << ": " << error.what() << '\n';
    return 1;
}
