#include "r10_catalog_source.h"

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

namespace {
using namespace r10cat;
using Clock = std::chrono::steady_clock;
constexpr int chunk = 256;
constexpr llama_seq_id scratch = 3;

void decode(llama_context *ctx, llama_seq_id seq, const Tokens &tokens, int pos, int end) {
    for (int offset = pos; offset < end; offset += chunk) {
        const int count = std::min(chunk, end - offset);
        auto batch = llama_batch_init(count, 0, 1);
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = tokens[offset + i];
            batch.pos[i] = offset + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq;
            batch.logits[i] = offset + i + 1 == end;
        }
        const int status = llama_decode(ctx, batch);
        llama_batch_free(batch);
        need(status == 0, "resident prefill decode failed");
        llama_synchronize(ctx);
    }
}

long long elapsed_us(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}
}

int main(int argc, char **argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    need(argc == 9, "usage: r11_resident_pair_probe MODEL DLLDIR DEVICE DEEP SUFFIX REPEATS SEED CTX");
    const int deep = std::stoi(argv[4]), suffix = std::stoi(argv[5]), reps = std::stoi(argv[6]);
    const int context = std::stoi(argv[8]);
    need(deep > 128 && deep % 64 == 0 && suffix > 0 && reps > 0 && deep + suffix + 128 <= context,
         "invalid pair geometry or context");
    std::mt19937 rng(std::stoul(argv[7]));
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devices[]{ggml_backend_dev_by_name(argv[3]), nullptr};
    need(devices[0], "native Vulkan GPU missing");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    need(bool(model), "load model failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = context;
    cp.n_batch = cp.n_ubatch = chunk;
    cp.n_seq_max = 4;
    cp.kv_unified = true;
    const std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    need(bool(ctx), "allocate context failed");
    const auto memory = llama_get_memory(ctx.get());
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    Tokens tokens(deep + suffix);
    for (auto &token : tokens) token = 1000 + rng() % 14000;
    const std::array<int, 3> depth{deep - 128, deep - 64, deep};
    decode(ctx.get(), 2, tokens, 0, depth[0]);
    llama_memory_seq_cp(memory, 2, 0, -1, -1);
    decode(ctx.get(), 2, tokens, depth[0], depth[1]);
    llama_memory_seq_cp(memory, 2, 1, -1, -1);
    decode(ctx.get(), 2, tokens, depth[1], depth[2]);
    std::cout << "rep,order,boundary,total,clone_us,prefill_us,greedy\n";
    for (int rep = -2; rep < reps; ++rep) {
        std::array<llama_seq_id, 3> order{0, 1, 2};
        std::shuffle(order.begin(), order.end(), rng);
        for (int j = 0; j < 3; ++j) {
            const auto source = order[j];
            for (int d = 0; d < 3; ++d)
                need(llama_memory_seq_pos_max(memory, d) == depth[d] - 1, "resident donor changed");
            const auto clone_start = Clock::now();
            llama_memory_seq_cp(memory, source, scratch, -1, -1);
            const auto clone = elapsed_us(clone_start);
            need(llama_memory_seq_pos_max(memory, scratch) == depth[source] - 1, "resident copy length differs");
            const auto start = Clock::now();
            decode(ctx.get(), scratch, tokens, depth[source], deep + suffix);
            const auto prefill = elapsed_us(start);
            need(llama_memory_seq_pos_max(memory, scratch) == deep + suffix - 1, "continuation length differs");
            const float *logits = llama_get_logits_ith(ctx.get(), -1);
            need(logits && std::all_of(logits, logits + vocab, [](float x) { return std::isfinite(x); }), "invalid logits");
            const auto greedy = std::max_element(logits, logits + vocab) - logits;
            need(llama_memory_seq_rm(memory, scratch, -1, -1), "scratch sequence cleanup failed");
            if (rep >= 0)
                std::cout << rep << ',' << j << ',' << depth[source] << ',' << deep + suffix << ',' << clone << ','
                          << prefill << ',' << greedy << std::endl;
        }
    }
    return 0;
} catch (const std::exception &error) {
    std::cerr << "R11 resident pair: " << error.what() << '\n';
    return 1;
}
