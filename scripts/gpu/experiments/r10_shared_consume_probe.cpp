#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
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

namespace {
using Tokens = std::vector<llama_token>;
using Bytes = std::vector<uint8_t>;
using Clock = std::chrono::steady_clock;
constexpr int suffix = 128;
constexpr int follow = 8;

void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

void decode(llama_context *ctx, llama_seq_id seq, llama_pos pos, const llama_token *data, int n) {
    auto batch = llama_batch_init(n, 0, 1);
    struct Free { llama_batch b; ~Free() { llama_batch_free(b); } } free{batch};
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = data[i];
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = i == n - 1;
    }
    check(llama_decode(ctx, batch) == 0, "decode failed");
    llama_synchronize(ctx);
}

void prefix(llama_context *ctx, const Tokens &tokens) {
    for (int pos = 0; pos < int(tokens.size()); pos += 256) {
        decode(ctx, 0, pos, tokens.data() + pos, 256);
    }
}

Bytes save(llama_context *ctx, llama_seq_id seq) {
    Bytes out(llama_state_seq_get_size(ctx, seq));
    check(out.size() > 8 && llama_state_seq_get_data(ctx, out.data(), out.size(), seq) == out.size(),
          "state readback failed");
    // The pinned Qwen3.5 unified M-RoPE encoding identifies the sequence in
    // its header and in every KV cell record. Keep positions, extended cell
    // metadata, and all tensor bytes; canonicalize only those owner IDs.
    auto u32 = [&out](size_t p) {
        uint32_t value = 0;
        check(p + sizeof(value) <= out.size(), "state metadata is truncated");
        std::memcpy(&value, out.data() + p, sizeof(value));
        return value;
    };
    check(u32(4) == uint32_t(seq) && u32(8) == 1, "unexpected sequence/stream metadata");
    std::fill(out.begin() + 4, out.begin() + 8, 0);
    const uint32_t count = u32(12);
    check(count == uint32_t(llama_memory_seq_pos_max(llama_get_memory(ctx), seq) + 1), "unexpected cell count");
    constexpr size_t metadata_bytes = 24; // pos, owner count, M-RoPE ext (x, y, token), owner
    for (uint32_t i = 0; i < count; ++i) {
        const auto start = 16 + metadata_bytes * i;
        check(u32(start) == i && u32(start + 4) == 1 && u32(start + 20) == uint32_t(seq),
              "unexpected KV position or owner; cannot compare this format");
        std::fill(out.begin() + start + 20, out.begin() + start + 24, 0);
    }
    return out;
}

using Logits = std::array<std::vector<float>, follow>;
Logits continue_tokens(llama_context *ctx, llama_seq_id seq, int depth, const Tokens &tokens, int vocab) {
    decode(ctx, seq, depth, tokens.data(), suffix);
    Logits out;
    for (int i = 0; i < follow; ++i) {
        const auto *raw = llama_get_logits_ith(ctx, -1);
        check(raw && std::all_of(raw, raw + vocab, [](float x) { return std::isfinite(x); }), "nonfinite logits");
        out[i].assign(raw, raw + vocab);
        if (i + 1 < follow) decode(ctx, seq, depth + suffix + i, tokens.data() + suffix + i, 1);
    }
    return out;
}

struct Diff { int greedy = 0; double maximum = 0; };
Diff diff(const Logits &a, const Logits &b) {
    Diff out;
    for (size_t i = 0; i < a.size(); ++i) {
        out.greedy += std::max_element(a[i].begin(), a[i].end()) - a[i].begin() ==
                      std::max_element(b[i].begin(), b[i].end()) - b[i].begin();
        for (size_t j = 0; j < a[i].size(); ++j) out.maximum = std::max(out.maximum, std::abs(double(a[i][j]) - b[i][j]));
    }
    return out;
}
}

int main(int argc, char **argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    check(argc == 6, "usage: r10_shared_consume_probe MODEL BACKENDS DEVICE REPEATS SEED");
    const int reps = std::stoi(argv[4]);
    check(reps > 0, "positive repeats required");
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name(argv[3]), nullptr};
    check(devices[0], "GPU backend missing");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    check(bool(model), "model load failed");
    const auto rope = llama_model_rope_type(model.get());
    check(rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE, "probe expects M-RoPE cell metadata");
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_seq_max = 3;
    cp.kv_unified = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> owner(llama_init_from_model(model.get(), cp), llama_free);
    check(bool(owner), "context init failed");
    auto *ctx = owner.get();
    const auto memory = llama_get_memory(ctx);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::mt19937 rng(std::stoul(argv[5]));
    auto randomize = [&rng](Tokens &tokens) { for (auto &t : tokens) t = 1000 + rng() % 14000; };
    std::cout << "rep,depth,shape,prefix_bytes,copy_metadata_ms,check_after_donor_release,check_after_other_writes,"
                 "check_other_after_write,greedy_first,max_error_first,greedy_second,max_error_second\n" << std::setprecision(10);
    for (int rep = -1; rep < reps; ++rep) {
        for (const int depth : {512, 1024, 1536}) {
            Tokens p(depth), a(suffix + follow - 1), b(suffix + follow - 1);
            randomize(p); randomize(a); randomize(b);
            llama_memory_clear(memory, false);
            prefix(ctx, p);
            const auto reference_a = continue_tokens(ctx, 0, depth, a, vocab);
            llama_memory_clear(memory, false);
            prefix(ctx, p);
            const auto reference_b = continue_tokens(ctx, 0, depth, b, vocab);
            llama_memory_clear(memory, false);
            prefix(ctx, p);
            const auto stored = save(ctx, 0);
            const auto before = Clock::now();
            llama_memory_seq_cp(memory, 0, 1, -1, -1);
            llama_memory_seq_cp(memory, 0, 2, -1, -1);
            const double copy_time = std::chrono::duration<double, std::milli>(Clock::now() - before).count();
            check(llama_memory_seq_rm(memory, 0, -1, -1) && llama_memory_seq_pos_max(memory, 0) < 0,
                  "donor is still live");
            check(llama_memory_seq_pos_max(memory, 1) == depth - 1 && llama_memory_seq_pos_max(memory, 2) == depth - 1,
                  "lost shared state after donor release");
            const bool same_after_donor = save(ctx, 1) == stored && save(ctx, 2) == stored;
            const auto shared_a = continue_tokens(ctx, 1, depth, a, vocab);
            const auto advanced_a = save(ctx, 1);
            const bool same_after_first = save(ctx, 2) == stored;
            const auto shared_b = continue_tokens(ctx, 2, depth, b, vocab);
            const bool same_after_second = save(ctx, 1) == advanced_a;
            const auto d1 = diff(reference_a, shared_a), d2 = diff(reference_b, shared_b);
            std::cout << rep << ',' << depth << ",resident-shared," << stored.size() << ',' << copy_time << ','
                      << same_after_donor << ',' << same_after_first << ',' << same_after_second << ','
                      << d1.greedy << ',' << d1.maximum << ',' << d2.greedy << ',' << d2.maximum << std::endl;
            check(same_after_donor && same_after_first && same_after_second, "resident shared state or branch isolation changed");
        }
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
