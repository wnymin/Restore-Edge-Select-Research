#include "r10_catalog_transfer.h"
#include "r10_catalog_witness.h"

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <thread>
#include <psapi.h>

namespace {
using namespace r10cat;
using Clock = std::chrono::steady_clock;
constexpr size_t MiB = 1 << 20, chunk = 256;
constexpr std::array<size_t, 4> sizes{8 * MiB, 16 * MiB, 32 * MiB, 52690944};
constexpr std::array<const char *, 4> arms{"compute-a", "compute-b", "transfer-only", "joint"};

LONG WINAPI fault(EXCEPTION_POINTERS *ex) {
    std::fprintf(stderr, "r10_joint_grid_fault,0x%08lx\n", ex->ExceptionRecord->ExceptionCode);
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}
int64_t qpc(bool frequency = false) {
    LARGE_INTEGER result{};
    need(frequency ? QueryPerformanceFrequency(&result) : QueryPerformanceCounter(&result), "performance counter unavailable");
    return result.QuadPart;
}
void resources(ggml_backend_dev_t dev, const char *phase) {
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    PROCESS_MEMORY_COUNTERS_EX mem{};
    need(K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&mem), sizeof(mem)),
         "could not read process memory counters");
    std::cerr << "r10_resources," << phase << ',' << free << ',' << total << ',' << mem.WorkingSetSize << ','
              << mem.PeakWorkingSetSize << ',' << mem.PrivateUsage << '\n';
}
struct ShapeCase { size_t depth, suffix, object; bool fit; };
struct Sample {
    int64_t cstart = -1, csubmitted = -1, cready = -1, xstart = -1, xsubmitted = -1, xready = -1;
    int64_t disk_us = 0;
    uint64_t xticket = 0, cvalue = 0;
    unsigned pending_at_xsubmit = 0, pending_at_xready = 0;
};
class Grid {
public:
    Grid(llama_context *ctx, ggml_backend_dev_t dev, const fs::path &root, const fs::path &observer, unsigned seed) :
        ctx_(ctx), random_(seed), plan_(transport_plan()), pool_(64 * MiB, {sizes.begin(), sizes.end()}),
        lane_(dev, plan_, root, pool_.capacity()), witness_(observer), marker_(ggml_backend_event_new(dev), ggml_backend_event_free) {
        need(bool(marker_), "missing native compute marker");
        tokens_.resize(1536);
        for (auto &t : tokens_) t = 300 + random_() % 49999;
        for (size_t i = 0; i < 768; i += chunk) {
            decode(i, chunk);
            if (i + chunk >= 512) {
                const auto size = llama_state_seq_get_size_ext(ctx_, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
                auto &data = states_[i + chunk];
                data.resize(size);
                need(size && llama_state_seq_get_data_ext(ctx_, data.data(), size, 0, LLAMA_STATE_SEQ_FLAGS_NONE) == size,
                     "cannot capture grid prefix");
            }
        }
    }
    void run(int reps, size_t vocab) {
        std::vector<ShapeCase> matrix;
        for (size_t object : {size_t(1), size_t(3)}) {
            for (size_t n : {size_t(64), size_t(512)}) matrix.push_back({512, n, object, true});
            for (size_t n : {size_t(128), size_t(384)}) matrix.push_back({768, n, object, false});
        }
        const int64_t frequency = qpc(true);
        std::cout << "repeat,position,suffix,tier,bytes,split,order,arm,qpc_hz,compute_start,compute_submitted,compute_ready,"
                     "transfer_start,transfer_submitted,transfer_ready,disk_us,dma_ticket,compute_timeline,"
                     "compute_pending_after_dma_submit,compute_pending_after_dma_ready\n";
        for (int rep = -1; rep < reps; ++rep) {
            std::shuffle(matrix.begin(), matrix.end(), random_);
            for (auto shape : matrix) {
                std::array<size_t, arms.size()> order{0, 1, 2, 3};
                std::shuffle(order.begin(), order.end(), random_);
                std::vector<float> expected;
                for (size_t i = 0; i < order.size(); ++i) {
                    const auto arm = order[i];
                    Sample s;
                    try {
                        s = sample(shape, arm != 2, arm >= 2, rep < 0);
                    } catch (const std::exception &e) {
                        throw std::runtime_error("repetition " + std::to_string(rep) + ", position " + std::to_string(shape.depth) +
                            ", suffix " + std::to_string(shape.suffix) + ", object " + std::to_string(shape.object) +
                            ", arm " + arms[arm] + ": " + e.what());
                    }
                    if (arm != 2) {
                        const auto *l = llama_get_logits_ith(ctx_, -1);
                        need(l && std::all_of(l, l + vocab, [](float x) {return std::isfinite(x);}), "nonfinite comparison logits");
                        if (expected.empty()) expected.assign(l, l + vocab);
                        else need(std::memcmp(expected.data(), l, vocab * sizeof(float)) == 0, "overlap changed compute logits");
                    }
                    if (rep >= 0) std::cout << rep << ',' << shape.depth << ',' << shape.suffix << ',' << (shape.object == 1 ? "RAM" : "SSD")
                          << ',' << sizes[shape.object] << ',' << (shape.fit ? "fit" : "validation") << ',' << i << ',' << arms[arm]
                          << ',' << frequency << ',' << s.cstart << ',' << s.csubmitted << ',' << s.cready
                          << ',' << s.xstart << ',' << s.xsubmitted << ',' << s.xready << ',' << s.disk_us
                          << ',' << s.xticket << ',' << s.cvalue << ',' << s.pending_at_xsubmit << ',' << s.pending_at_xready << '\n' << std::flush;
                }
            }
        }
    }
    size_t resident() const { return lane_.resident_ram_bytes(); }
    size_t pinned() const { return lane_.pinned_bytes(); }
    size_t arena() const { return lane_.arena_bytes(); }
private:
    static Plan transport_plan() {
        Plan p;
        for (size_t i = 0; i < sizes.size(); ++i) p.objects.push_back({i == 3 ? 'R' : 'K', sizes[i], {}});
        return p;
    }
    void decode(size_t start, size_t n) {
        need(n && n <= chunk && llama_decode(ctx_, llama_batch_get_one(tokens_.data() + start, int(n))) == 0, "grid decode failed");
    }
    void reset(size_t depth) {
        const auto &ref = states_.at(depth);
        llama_memory_clear(llama_get_memory(ctx_), false);
        need(llama_state_seq_set_data_ext(ctx_, ref.data(), ref.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) == ref.size(), "cannot restore grid prefix");
        llama_synchronize(ctx_);
    }
    Sample sample(ShapeCase sh, bool compute, bool transfer, bool warmup) {
        if (compute) reset(sh.depth);
        if (transfer) need(pool_.reserve(0, {sh.object}), "solo staged object exceeds arena");
        const auto deadline = Clock::now() + std::chrono::seconds(warmup ? 40 : 10);
        const int64_t origin = qpc();
        const auto at = [&]() { return qpc() - origin; };
        Sample s;
        bool active = false;
        size_t pos = sh.depth;
        const auto end = sh.depth + sh.suffix;
        const auto launch = [&]() {
            const size_t count = std::min(chunk, end - pos);
            decode(pos, count);
            pos += count;
            witness_.observe(++id_, [&] { ggml_backend_event_record(marker_.get(), lane_.compute_marker_backend()); });
            active = true;
        };
        if (compute) {
            s.cstart = at();
            launch();
            s.csubmitted = witness_.poll(id_).second.submitted_qpc - origin;
        }
        const auto observe = [&](const DeviceTransfer::Step &step) {
            if (step.ssd_bytes) {
                need(step.ssd_bytes == sizes[sh.object], "incomplete SSD bytes in overlap");
                s.disk_us = step.ssd_observation_us;
            }
            if (step.submitted) {
                s.xsubmitted = at();
                s.pending_at_xsubmit = unsigned(active && !witness_.poll(id_).first);
                s.xticket = step.submitted;
                pool_.issued(sh.object, step.submitted);
            }
            if (step.completed) {
                s.xready = at();
                s.pending_at_xready = unsigned(active && !witness_.poll(id_).first);
                need(s.xticket == step.completed, "DMA completion identity mismatch");
                pool_.landed(sh.object, step.completed);
            }
        };
        if (transfer) {
            s.xstart = at();
            observe(lane_.read(sh.object, nullptr, pool_));
        }
        while (active || lane_.busy()) {
            if (lane_.busy()) observe(lane_.poll(pool_));
            if (active) {
                const auto [ready, record] = witness_.poll(id_);
                if (ready && pos == end) {
                    s.cready = record.observed_ready_qpc - origin;
                    s.cvalue = record.timeline_value;
                    active = false;
                } else if (ready) launch();
            }
            if (active || lane_.busy()) {
                if (Clock::now() >= deadline) throw std::runtime_error("deadline with compute=" + std::to_string(active) +
                    ", transport=" + std::to_string(lane_.busy()) + ", last submitted position=" + std::to_string(pos) +
                    ", ticket=" + std::to_string(s.xticket));
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        if (transfer) {
            need(pool_.consumable(0), "overlap ended without a consumable transfer");
            pool_.release(0);
        }
        return s;
    }
    llama_context *ctx_;
    std::mt19937 random_;
    Plan plan_;
    R10StagePool pool_;
    DeviceTransfer lane_;
    TimelineWitness witness_;
    std::unique_ptr<ggml_backend_event, decltype(&ggml_backend_event_free)> marker_;
    Tokens tokens_;
    std::map<size_t, Bytes> states_;
    uint64_t id_ = 0;
};
}

int main(int argc, char **argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    SetUnhandledExceptionFilter(fault);
    need(argc == 8, "usage: r10_joint_grid MODEL DLLDIR DEVICE TRANSFER_OBJECTS OBSERVER REPEATS SEED");
    const auto reps = std::stoi(argv[6]);
    need(reps > 0 && reps <= 30, "invalid repetition count");
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devices[]{ggml_backend_dev_by_name(argv[3]), nullptr};
    need(devices[0], "joint grid GPU unavailable");
    resources(devices[0], "before-model");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    need(bool(model), "cannot load joint grid model");
    resources(devices[0], "after-model");
    auto cp = llama_context_default_params();
    cp.n_ctx = 3072;
    cp.n_batch = cp.n_ubatch = chunk;
    cp.n_seq_max = 1;
    const std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    need(bool(ctx), "cannot load joint grid context");
    resources(devices[0], "after-context");
    Grid grid(ctx.get(), devices[0], argv[4], argv[5], std::stoul(argv[7]));
    resources(devices[0], "after-grid-preparation");
    std::cerr << "r10_grid_buffers," << grid.arena() << ',' << grid.resident() << ',' << grid.pinned() << '\n';
    grid.run(reps, size_t(llama_vocab_n_tokens(llama_model_get_vocab(model.get()))));
    resources(devices[0], "after-timed-grid");
    return 0;
} catch (const std::exception &e) {
    std::cerr << "r10 joint grid: " << e.what() << '\n';
    return 1;
}
