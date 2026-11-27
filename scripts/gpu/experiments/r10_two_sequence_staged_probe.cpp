// Qualification only: two live sequences restored through one R10 device arena.
#include "r10_catalog_transfer.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>

namespace {
using namespace r10cat;
using Clock = std::chrono::steady_clock;
constexpr int depth = 512, chunk = 256, tail = 64, steps = 8;

struct Batch {
    llama_batch rows = llama_batch_init(chunk, 0, 1);
    ~Batch() { llama_batch_free(rows); }
    void add(llama_token token, int pos, llama_seq_id seq, bool output = true) {
        need(rows.n_tokens < chunk, "fixture batch is too large");
        const int i = rows.n_tokens++;
        rows.token[i] = token;
        rows.pos[i] = pos;
        rows.n_seq_id[i] = 1;
        rows.seq_id[i][0] = seq;
        rows.logits[i] = output;
    }
    void run(llama_context *ctx) {
        need(llama_decode(ctx, rows) == 0, "fixture decode failed");
        llama_synchronize(ctx);
    }
};

void append(llama_context *ctx, const Tokens &tokens, int first, int last, llama_seq_id seq) {
    Batch batch;
    for (int p = first; p < last; ++p) batch.add(tokens.at(p), p, seq, p + 1 == last);
    batch.run(ctx);
}

void position(llama_context *ctx, llama_seq_id seq, llama_pos expected) {
    need(llama_memory_seq_pos_max(llama_get_memory(ctx), seq) == expected, "sequence position is wrong");
}

Bytes save(llama_context *ctx, llama_seq_id seq) {
    Bytes state(llama_state_seq_get_size(ctx, seq));
    need(!state.empty() && llama_state_seq_get_data(ctx, state.data(), state.size(), seq) == state.size(), "cannot serialize live sequence");
    return state;
}

Target bind_empty(llama_context *ctx, const Meta &meta, llama_seq_id seq, ggml_backend_buffer_type_t type) {
    Shape spans{};
    size_t n = 0;
    position(ctx, seq, -1);
    need(llama_r10_prepare_seq_restore(ctx, meta.skeleton.data(), meta.skeleton.size(), seq, spans.data(), spans.size(), &n)
          == meta.skeleton.size() && n == meta.regions.size(), "cannot plan staged restore into an empty sequence");
    position(ctx, seq, depth - 1);
    Target target;
    for (size_t i = 0; i < n; ++i) {
        const auto &s = spans[i];
        const auto &r = meta.regions.at(i);
        // At depth 512 each tensor span is entirely within either the KV or recurrent object.
        need(r.parts.size() == 1 && s.tensor && s.tensor->buffer && s.tensor->data && s.size == r.bytes &&
             r.tensor == ggml_get_name(s.tensor) && s.src == meta.skeleton.data() + r.source &&
             ggml_backend_buffer_get_type(s.tensor->buffer) == type, "unexpected sealed tensor span for target sequence");
        const auto &p = r.parts.front();
        const auto base = reinterpret_cast<uintptr_t>(s.tensor->data);
        need(s.offset <= std::numeric_limits<uintptr_t>::max() - base && p.bytes == s.size &&
             s.size <= std::numeric_limits<uintptr_t>::max() - base - s.offset, "invalid native target address range");
        target[p.object].push_back({{s.tensor->buffer, base + s.offset, p.bytes, p.offset}, s.tensor, s.offset});
    }
    return target;
}

void transfer_one(DeviceTransfer &io, R10StagePool &pool, size_t id) {
    const auto cutoff = Clock::now() + std::chrono::seconds(15);
    auto status = io.read(id, nullptr, pool);
    size_t ssd_bytes = 0;
    while (true) {
        ssd_bytes += status.ssd_bytes;
        if (status.submitted) {
            pool.issued(id, status.submitted);
            need(!pool.consumable(0) && !pool.consumable(1), "a lease became ready before its device fence");
            std::cout << "issued," << id << ',' << status.submitted << ',' << pool.charged() << ','
                      << pool.consumable(0) << ',' << pool.consumable(1) << '\n';
        }
        if (status.completed) {
            pool.landed(id, status.completed);
            std::cout << "landed," << id << ',' << status.completed << ',' << pool.get(id).bytes << ',' << ssd_bytes << '\n';
            return;
        }
        need(Clock::now() < cutoff, "timed out waiting for the device transfer ticket");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        status = io.poll(pool);
    }
}

void restore(llama_context *ctx, const Meta &meta, const Candidate &c, DeviceTransfer &io, R10StagePool &pool, llama_seq_id seq) {
    need(pool.matches(seq, c.deps) && pool.consumable(seq), "attempted restore without the exact ready lease");
    const auto native = bind_empty(ctx, meta, seq, io.device_type());
    size_t bytes = 0;
    for (const auto id : c.deps) bytes += io.device_copy(id, native, pool, false);
    pool.release(seq);
    std::cout << "copied," << seq << ',' << bytes << ',' << pool.charged() << '\n';
}

void capture(llama_context *ctx, int row, int vocab, std::vector<float> &dst) {
    const float *p = llama_get_logits_ith(ctx, row);
    need(p && std::all_of(p, p + vocab, [](float v) { return std::isfinite(v); }), "nonfinite or missing logits");
    dst.insert(dst.end(), p, p + vocab);
}

struct Result { std::vector<float> a, b; Bytes before_b_a, ready_b, final_a, final_b; };

Bytes sealed_image(const Plan &plan, const fs::path &root, const Candidate &candidate, const Meta &meta) {
    std::map<size_t, Bytes> objects;
    for (const auto id : candidate.deps) {
        auto &bytes = objects[id];
        bytes.resize(plan.objects.at(id).bytes);
        std::ifstream file(path_for(root, id, false), std::ios::binary);
        need(bool(file.read(reinterpret_cast<char *>(bytes.data()), bytes.size())), "cannot read sealed object for synchronous control");
    }
    auto state = meta.skeleton;
    for (const auto &region : meta.regions) {
        auto offset = region.source;
        for (const auto &part : region.parts) {
            const auto &source = objects.at(part.object);
            need(part.offset <= source.size() && part.bytes <= source.size() - part.offset &&
                 offset <= state.size() && part.bytes <= state.size() - offset, "sealed control span outside object or state");
            std::memcpy(state.data() + offset, source.data() + part.offset, part.bytes);
            offset += part.bytes;
        }
        need(offset - region.source == region.bytes, "incomplete sealed control span");
    }
    return state;
}

void reject_one_page_short(const Candidate &c, const std::vector<size_t> &sizes, size_t capacity) {
    R10StagePool too_small(capacity - R10StagePool::page, sizes);
    const bool rejected = !too_small.reserve(0, c.deps);
    need(rejected && !too_small.charged(), "one-page-short arena incorrectly admitted the union");
    std::cout << "capacity-reject," << too_small.capacity() << ',' << too_small.charged() << ',' << rejected << '\n';
}

void stage_shared(const Candidate &c, DeviceTransfer &io, R10StagePool &pool, size_t capacity) {
    need(pool.reserve(0, c.deps) && pool.charged() == capacity && !pool.consumable(0), "initial lease or pre-fence gate failed");
    // This candidate starts with a RAM KV object: submit before joining the second lease.
    const auto first = c.deps.front();
    const auto status = io.read(first, nullptr, pool);
    need(status.submitted && !status.completed, "first resident KV upload did not return a device ticket");
    pool.issued(first, status.submitted);
    const auto last = c.deps.back();
    need(pool.reserve(1, c.deps) && pool.charged() == capacity && pool.get(first).refs == 2 && pool.get(last).refs == 2 &&
         !pool.consumable(0) && !pool.consumable(1), "joining an in-flight lease charged duplicates or exposed unready data");
    std::cout << "join-flying," << first << ',' << status.submitted << ',' << pool.charged() << ','
              << pool.get(first).refs << ',' << pool.get(last).refs << ',' << pool.consumable(0) << ',' << pool.consumable(1) << '\n';
    const auto cutoff = Clock::now() + std::chrono::seconds(15);
    uint64_t done = 0;
    while (!(done = io.poll(pool).completed)) {
        need(Clock::now() < cutoff, "timed out waiting for the shared KV upload");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    pool.landed(first, done);
    need(!pool.consumable(0) && !pool.consumable(1), "one of two dependencies incorrectly made the full lease ready");
    std::cout << "landed," << first << ',' << done << ',' << pool.get(first).bytes << ",0\n";
    for (size_t i = 1; i < c.deps.size(); ++i) transfer_one(io, pool, c.deps[i]);
}

void reject_occupied(llama_context *ctx, const Meta &meta, const Bytes &live) {
    Shape invalid{};
    size_t n = invalid.size();
    const auto used = llama_r10_prepare_seq_restore(ctx, meta.skeleton.data(), meta.skeleton.size(), 0, invalid.data(), invalid.size(), &n);
    const bool kept = save(ctx, 0) == live;
    need(!used && !n && kept, "restore into the occupied sequence did not fail without mutation");
    std::cout << "occupied-reject,0," << used << ',' << n << ',' << kept << '\n';
}

Bytes restore_shared(llama_context *ctx, const Tokens &a, const Candidate &c, const Meta &meta,
                     const std::vector<size_t> &sizes, size_t capacity, DeviceTransfer &io) {
    reject_one_page_short(c, sizes, capacity);
    R10StagePool pool(capacity, sizes);
    stage_shared(c, io, pool, capacity);
    restore(ctx, meta, c, io, pool, 0);
    need(pool.charged() == capacity && pool.consumable(1), "first consumer released the remaining lease's pages");
    append(ctx, a, depth, depth + 1, 0);
    const auto live = save(ctx, 0);
    reject_occupied(ctx, meta, live);
    restore(ctx, meta, c, io, pool, 1);
    const bool isolated = save(ctx, 0) == live;
    const auto first = c.deps.front(), last = c.deps.back();
    need(!pool.charged() && isolated && !pool.get(first).refs && !pool.get(last).refs,
         "second restore changed the live sequence or leaked arena pages");
    std::cout << "isolated," << isolated << ',' << pool.charged() << ',' << pool.get(first).refs << ',' << pool.get(last).refs << '\n';
    return live;
}

Result trial(llama_context *ctx, const Tokens &a, const Tokens &b, int vocab, const Candidate &c, const Meta &meta,
             const std::vector<size_t> &sizes, size_t capacity, DeviceTransfer &io, bool staged, bool b_first, const Bytes &reference) {
    Result result;
    llama_memory_clear(llama_get_memory(ctx), true);
    if (staged) {
        result.before_b_a = restore_shared(ctx, a, c, meta, sizes, capacity, io);
    } else {
        append(ctx, a, 0, chunk, 0);
        append(ctx, a, chunk, depth, 0);
        append(ctx, a, depth, depth + 1, 0);
        result.before_b_a = save(ctx, 0);
        if (reference.empty()) {
            append(ctx, b, 0, chunk, 1);
            append(ctx, b, chunk, depth, 1);
        } else {
            need(llama_state_seq_set_data(ctx, reference.data(), reference.size(), 1) == reference.size(),
                 "cannot synchronously restore the same sealed checkpoint while A is live");
            need(save(ctx, 0) == result.before_b_a, "synchronous reference changed live sequence A");
        }
    }
    result.ready_b = save(ctx, 1);
    position(ctx, 0, depth);
    position(ctx, 1, depth - 1);
    Batch mixed;
    if (!b_first) mixed.add(a.at(depth + 1), depth + 1, 0);
    for (int i = 0; i < tail; ++i) mixed.add(b.at(depth + i), depth + i, 1, i + 1 == tail);
    if (b_first) mixed.add(a.at(depth + 1), depth + 1, 0);
    std::cerr << "probe-mixed-begin saved-b=" << staged << " b-first=" << b_first << '\n';
    mixed.run(ctx);
    std::cerr << "probe-mixed-end\n";
    capture(ctx, b_first ? tail : 0, vocab, result.a);
    capture(ctx, b_first ? tail - 1 : tail, vocab, result.b);
    for (int i = 0; i < steps; ++i) {
        Batch dual;
        dual.add(a.at(depth + 2 + i), depth + 2 + i, 0);
        dual.add(b.at(depth + tail + i), depth + tail + i, 1);
        dual.run(ctx);
        capture(ctx, 0, vocab, result.a);
        capture(ctx, 1, vocab, result.b);
    }
    result.final_a = save(ctx, 0);
    result.final_b = save(ctx, 1);
    return result;
}
}

int main(int argc, char **argv) try {
    const bool shared = argc >= 7 && std::string(argv[6]) == "--shared-kv-stream";
    const bool sealed = argc == 8 && std::string(argv[7]) == "--sealed-reference";
    need(argc == 6 || (shared && (argc == 7 || sealed)),
         "usage: r10_two_sequence_staged_probe MODEL RUNTIME DEVICE PLAN CATALOGUE [--shared-kv-stream [--sealed-reference]]");
    const auto plan = read_plan(argv[4]);
    const auto fixture = std::find_if(plan.requests.begin(), plan.requests.end(), [](const Row &r) { return r.options.count(depth); });
    need(fixture != plan.requests.end(), "calibration catalogue lacks a 512-token checkpoint");
    const auto &row = *fixture;
    const auto &candidate = row.options.at(depth);
    need(candidate.deps.size() == 2 && plan.objects.at(candidate.deps.front()).kind == 'K' &&
         plan.objects.at(candidate.deps.back()).kind == 'R', "fixture requires one RAM KV and one SSD recurrent object");
    const auto meta = load_meta(path_for(argv[5], candidate.deps.back(), true), plan, candidate);
    std::vector<size_t> sizes;
    for (const auto &obj : plan.objects) sizes.push_back(obj.bytes);
    size_t capacity = 0;
    for (const auto id : candidate.deps) capacity += R10StagePool::cost(sizes.at(id));
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devices[]{ggml_backend_dev_by_name(argv[3]), nullptr};
    need(devices[0] && ggml_backend_dev_type(devices[0]) == GGML_BACKEND_DEVICE_TYPE_GPU, "expected fixed native GPU");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    need(model && llama_model_is_hybrid(model.get()), "expected hybrid model");
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = chunk;
    cp.n_seq_max = 2;
    cp.kv_unified = shared;
    const std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    need(ctx && llama_n_seq_max(ctx.get()) == 2 && llama_n_ctx_seq(ctx.get()) >= depth + tail + steps, "missing two native sequence slots");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    need(vocab > 4096, "unexpected model vocabulary");
    Tokens a(row.tokens.begin(), row.tokens.begin() + depth), b = a;
    for (int i = 0; i < tail + steps; ++i) { a.push_back(1024 + i * 7); b.push_back(2048 + i * 3); }
    const auto reference = sealed ? sealed_image(plan, argv[5], candidate, meta) : Bytes{};
    DeviceTransfer io(devices[0], plan, argv[5], capacity);
    std::cout << "arena," << io.arena_bytes() << ',' << candidate.deps.front() << ',' << candidate.deps.back() << std::endl;
    bool exact = true;
    for (const bool b_first : {false, true}) {
        const auto ref = trial(ctx.get(), a, b, vocab, candidate, meta, sizes, capacity, io, false, b_first, reference);
        const auto got = trial(ctx.get(), a, b, vocab, candidate, meta, sizes, capacity, io, true, b_first, reference);
        const auto equal = [](const auto &left, const auto &right) {
            return left.size() == right.size() && std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
        };
        const bool a_start = ref.before_b_a == got.before_b_a, b_start = ref.ready_b == got.ready_b;
        const bool a_end = ref.final_a == got.final_a, b_end = ref.final_b == got.final_b;
        const bool a_logits = equal(ref.a, got.a), b_logits = equal(ref.b, got.b);
        std::cout << "comparison," << b_first << ',' << a_start << ',' << b_start << ',' << a_end << ',' << b_end << ','
                  << a_logits << ',' << b_logits << ',' << ref.a.size() / vocab << std::endl;
        exact = exact && a_start && b_start && a_end && b_end && a_logits && b_logits;
    }
    need(exact, "staged two-sequence state differs under the same batch shape");
    return 0;
} catch (const std::exception &error) {
    std::cerr << "staged two-sequence qualification: " << error.what() << '\n';
    return 1;
}
