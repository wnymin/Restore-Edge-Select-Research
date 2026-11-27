// Check two live hybrid sequences and a mixed prefill/decode submission. No policy timing.
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
constexpr int prefix = 512, suffix = 64, probes = 8;
using Tokens = std::vector<llama_token>;
using Bytes = std::vector<std::uint8_t>;
using Logits = std::vector<float>;

void need(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

struct Batch {
    llama_batch value = llama_batch_init(prefix, 0, 1);
    ~Batch() { llama_batch_free(value); }

    void add(llama_token token, llama_pos pos, llama_seq_id seq, bool output) {
        const int i = value.n_tokens++;
        need(i < prefix, "fixture batch overflow");
        value.token[i] = token;
        value.pos[i] = pos;
        value.n_seq_id[i] = 1;
        value.seq_id[i][0] = seq;
        value.logits[i] = output;
    }
};

void decode(llama_context *ctx, Batch &batch) {
    need(llama_decode(ctx, batch.value) == 0, "llama_decode failed");
    llama_synchronize(ctx);
}

void append(llama_context *ctx, llama_seq_id id, const Tokens &tokens, int start, int end, bool output = false) {
    Batch batch;
    for (int i = start; i < end; ++i) batch.add(tokens.at(i), i, id, output && i + 1 == end);
    decode(ctx, batch);
}

void at(llama_context *ctx, llama_seq_id seq, llama_pos want) {
    need(llama_memory_seq_pos_max(llama_get_memory(ctx), seq) == want, "sequence positions disagree");
}

Bytes state(llama_context *ctx, llama_seq_id seq) {
    Bytes bytes(llama_state_seq_get_size(ctx, seq));
    need(!bytes.empty() && llama_state_seq_get_data(ctx, bytes.data(), bytes.size(), seq) == bytes.size(),
         "failed to save complete sequence checkpoint");
    return bytes;
}

Logits read_logits(llama_context *ctx, int row, int vocab) {
    const float *p = llama_get_logits_ith(ctx, row);
    need(p && std::all_of(p, p + vocab, [](float x) { return std::isfinite(x); }), "missing or non-finite logits");
    return Logits(p, p + vocab);
}

Tokens fixture(unsigned x, int n, int vocab) {
    Tokens tokens;
    for (int i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        tokens.push_back(llama_token(128 + (x >> 8) % unsigned(std::min(vocab - 128, 32768))));
    }
    return tokens;
}

struct Trial {
    std::vector<Logits> a, b;
    Bytes final_a, final_b;
};

Trial mixed(llama_context *ctx, const Tokens &a, const Tokens &b, int vocab, const Bytes *saved_b, bool swap_rows) {
    llama_memory_clear(llama_get_memory(ctx), true);
    append(ctx, 0, a, 0, prefix);
    at(ctx, 0, prefix - 1);
    at(ctx, 1, -1);
    if (saved_b) {
        const auto live_a = state(ctx, 0);
        need(llama_state_seq_set_data(ctx, saved_b->data(), saved_b->size(), 1) == saved_b->size(),
             "could not restore saved B into sequence 1 while A was live");
        need(state(ctx, 0) == live_a, "restoring B changed the serialized state of live A");
    } else {
        append(ctx, 1, b, 0, prefix);
    }
    at(ctx, 0, prefix - 1);
    at(ctx, 1, prefix - 1);
    Trial result;
    Batch mixed_batch;
    if (!swap_rows) mixed_batch.add(a.at(prefix), prefix, 0, true);
    for (int i = prefix; i < prefix + suffix; ++i) mixed_batch.add(b.at(i), i, 1, i + 1 == prefix + suffix);
    if (swap_rows) mixed_batch.add(a.at(prefix), prefix, 0, true);
    std::cerr << "probe-mixed-begin saved-b=" << bool(saved_b) << " b-first=" << swap_rows << '\n';
    decode(ctx, mixed_batch);
    std::cerr << "probe-mixed-end\n";
    result.a.push_back(read_logits(ctx, swap_rows ? suffix : 0, vocab));
    result.b.push_back(read_logits(ctx, swap_rows ? suffix - 1 : suffix, vocab));
    at(ctx, 0, prefix);
    at(ctx, 1, prefix + suffix - 1);
    for (int i = 0; i < probes; ++i) {
        Batch dual;
        dual.add(a.at(prefix + 1 + i), prefix + 1 + i, 0, true);
        dual.add(b.at(prefix + suffix + i), prefix + suffix + i, 1, true);
        decode(ctx, dual);
        result.a.push_back(read_logits(ctx, 0, vocab));
        result.b.push_back(read_logits(ctx, 1, vocab));
        at(ctx, 0, prefix + 1 + i);
        at(ctx, 1, prefix + suffix + i);
    }
    result.final_a = state(ctx, 0);
    result.final_b = state(ctx, 1);
    return result;
}

std::vector<Logits> solo(llama_context *ctx, llama_seq_id id, const Tokens &tokens, int trailing, int vocab) {
    llama_memory_clear(llama_get_memory(ctx), true);
    append(ctx, id, tokens, 0, prefix);
    append(ctx, id, tokens, prefix, prefix + trailing, true);
    std::vector<Logits> rows{read_logits(ctx, trailing - 1, vocab)};
    for (int i = prefix + trailing; i < int(tokens.size()); ++i) {
        append(ctx, id, tokens, i, i + 1, true);
        rows.push_back(read_logits(ctx, 0, vocab));
    }
    return rows;
}

void report(int order, const char *comparison, char seq, const std::vector<Logits> &ref, const std::vector<Logits> &measured,
            int state_exact, bool strict) {
    need(ref.size() == measured.size(), "incomplete logits comparison");
    int exact = 0, equal_top = 0;
    double largest = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const auto &x = ref[i], &y = measured[i];
        need(x.size() == y.size(), "mismatched logits vocabulary");
        exact += std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
        equal_top += (std::max_element(x.begin(), x.end()) - x.begin()) == (std::max_element(y.begin(), y.end()) - y.begin());
        for (std::size_t v = 0; v < x.size(); ++v) largest = std::max(largest, std::abs(double(x[v]) - double(y[v])));
    }
    std::cout << order << ',' << comparison << ',' << seq << ',' << ref.size() << ',' << exact << ',' << equal_top << ','
              << largest << ',' << state_exact << std::endl;
    if (strict) need(exact == int(ref.size()) && state_exact == 1, "restored mixed batch differs from the identical recomputed schedule");
}
}

int main(int argc, char **argv) try {
    need(argc == 4, "usage: r10_two_sequence_probe MODEL RUNTIME_DIR DEVICE");
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devices[]{ggml_backend_dev_by_name(argv[3]), nullptr};
    need(devices[0] && ggml_backend_dev_type(devices[0]) == GGML_BACKEND_DEVICE_TYPE_GPU, "requested GPU device unavailable");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    need(bool(model) && llama_model_is_hybrid(model.get()), "expected a loadable hybrid model");
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = prefix;
    cp.n_seq_max = 2;
    const std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
    need(bool(context) && llama_n_seq_max(context.get()) == 2 && llama_n_ctx_seq(context.get()) >= unsigned(prefix + suffix + probes),
         "could not allocate two usable sequences");
    auto *ctx = context.get();
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    need(vocab > 256, "unexpected vocabulary size");
    const auto a = fixture(32461, prefix + 1 + probes, vocab), b = fixture(92681, prefix + suffix + probes, vocab);
    append(ctx, 0, b, 0, prefix);
    const auto saved_b = state(ctx, 0);
    const auto solo_a = solo(ctx, 0, a, 1, vocab), solo_b = solo(ctx, 1, b, suffix, vocab);
    std::cout << "b_prefill_first,comparison,sequence,logit_positions,exact_positions,greedy_positions,max_abs_logit_diff,final_state_exact_or_na\n";
    for (int order = 0; order < 2; ++order) {
        const auto control = mixed(ctx, a, b, vocab, nullptr, order != 0);
        const auto restored = mixed(ctx, a, b, vocab, &saved_b, order != 0);
        report(order, "restore_vs_identical_mixed", 'A', control.a, restored.a, control.final_a == restored.final_a, true);
        report(order, "restore_vs_identical_mixed", 'B', control.b, restored.b, control.final_b == restored.final_b, true);
        report(order, "mixed_vs_serial", 'A', solo_a, control.a, -1, false);
        report(order, "mixed_vs_serial", 'B', solo_b, control.b, -1, false);
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << "two-sequence qualification: " << e.what() << '\n';
    return 1;
}
