#include "r10_catalog_source.h"

#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>

namespace {
using namespace r10cat;
using Clock = std::chrono::steady_clock;
constexpr size_t microbatch = 256, context = 3072;

size_t us(Clock::time_point t) {
    return size_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t).count());
}
LONG WINAPI fault(EXCEPTION_POINTERS *ex) {
    std::fprintf(stderr, "r10_prefill_grid_fault,0x%08lx\n", ex->ExceptionRecord->ExceptionCode);
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}
void decode(llama_context *ctx, Tokens &t, size_t offset, size_t n) {
    need(n && n <= microbatch && offset + n <= t.size() &&
         llama_decode(ctx, llama_batch_get_one(t.data() + offset, int(n))) == 0, "prefill grid decode failed");
}
Bytes save(llama_context *ctx) {
    Bytes state(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE));
    need(!state.empty() && llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) == state.size(), "could not capture cached prefill grid setup");
    return state;
}
void reset(llama_context *ctx, const std::map<size_t, Bytes> &cached, size_t depth) {
    llama_memory_clear(llama_get_memory(ctx), false);
    if (depth) {
        const auto &b = cached.at(depth);
        need(llama_state_seq_set_data_ext(ctx, b.data(), b.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) == b.size(), "could not restore cached grid setup");
    }
    llama_synchronize(ctx);
}
}

int main(int argc, char **argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    SetUnhandledExceptionFilter(fault);
    need(argc == 6, "usage: r10_prefill_grid MODEL DLLDIR DEVICE REPEATS SEED");
    const auto reps = std::stoul(argv[4]);
    need(reps > 0 && reps <= 50, "invalid repeat count");
    std::mt19937 random(std::stoul(argv[5]));
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devices[]{ggml_backend_dev_by_name(argv[3]), nullptr};
    need(devices[0], "could not find native GPU");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    need(bool(model), "could not load full grid model");
    auto cp = llama_context_default_params();
    cp.n_ctx = context;
    cp.n_batch = cp.n_ubatch = microbatch;
    cp.n_seq_max = 1;
    const std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    need(bool(ctx), "could not allocate grid context");
    Tokens prompt(context - 16);
    for (auto &t : prompt) t = 300 + random() % 49999;
    std::map<size_t, Bytes> cached;
    // Position before the chunk is sampled more frequently than legal recovery
    // boundaries: the evaluator can sum actual 256-token worker chunks.
    constexpr std::array<size_t, 6> train_positions{0, 512, 1024, 1536, 2048, 2560};
    constexpr std::array<size_t, 5> train_chunks{1, 16, 64, 128, 256};
    std::vector<std::pair<size_t, size_t>> shapes;
    for (size_t at : train_positions) for (size_t count : train_chunks) shapes.emplace_back(at, count);
    const std::vector<std::pair<size_t, size_t>> checks{{256, 32}, {768, 95}, {1280, 191}, {1792, 240}, {2304, 48}, {2304, 256}};
    shapes.insert(shapes.end(), checks.begin(), checks.end());
    for (size_t at = 0; at < train_positions.back(); at += microbatch) {
        decode(ctx.get(), prompt, at, microbatch);
        const auto depth = at + microbatch;
        if (std::any_of(shapes.begin(), shapes.end(), [depth](const auto &s) { return s.first == depth; })) cached.emplace(depth, save(ctx.get()));
    }
    std::cout << "repeat,order,position,tokens,split,submit_us,complete_us\n";
    for (int repetition = -1; repetition < int(reps); ++repetition) {
        std::shuffle(shapes.begin(), shapes.end(), random);
        for (size_t order = 0; order < shapes.size(); ++order) {
            const auto [depth, count] = shapes[order];
            reset(ctx.get(), cached, depth);
            const auto start = Clock::now();
            decode(ctx.get(), prompt, depth, count);
            const auto submitted = us(start);
            llama_synchronize(ctx.get());
            const auto done = us(start);
            need(done && done >= submitted, "nonpositive or reversed prefill timing");
            if (repetition >= 0) {
                std::cout << repetition << ',' << order << ',' << depth << ',' << count << ','
                          << (std::find(checks.begin(), checks.end(), shapes[order]) == checks.end() ? "fit" : "validation") << ','
                          << submitted << ',' << done << '\n' << std::flush;
            }
        }
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << "r10 prefill grid: " << e.what() << '\n';
    return 1;
}
