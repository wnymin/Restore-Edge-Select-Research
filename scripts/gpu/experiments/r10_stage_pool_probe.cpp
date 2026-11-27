#include "llama.h"
#include "ggml-backend.h"
#include "r10_stage_pool.h"
#include "r10_vk_trace_abi.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<uint8_t>;
using Tokens = std::vector<llama_token>;
using Shape = std::array<llama_r10_restore_span, 512>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Meta = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
constexpr size_t kv0 = 0, r512 = 1, kv1 = 2, r1024 = 3;
constexpr int n_steps = 8;
const std::vector<size_t> shallow{kv0, r512}, deep{kv0, kv1, r1024};
const char *volatile phase = "start";

LONG WINAPI fault(EXCEPTION_POINTERS *ex) {
    std::fprintf(stderr, "r10_stage_pool_fault,0x%08lx,phase=%s\n", ex->ExceptionRecord->ExceptionCode, phase);
    return EXCEPTION_EXECUTE_HANDLER;
}

void expect(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
size_t round_to(size_t n, size_t align) { return (n + align - 1) / align * align; }

void decode(llama_context *ctx, Tokens &tokens, size_t offset, size_t n) {
    expect(n && offset <= tokens.size() && n <= tokens.size() - offset, "invalid model input");
    expect(llama_decode(ctx, llama_batch_get_one(tokens.data() + offset, int(n))) == 0, "decode failed");
}

void clear(llama_context *ctx, bool payload) {
    llama_synchronize(ctx);
    llama_memory_clear(llama_get_memory(ctx), payload);
}

Bytes save(llama_context *ctx, llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    Bytes b(llama_state_seq_get_size_ext(ctx, 0, flags));
    expect(!b.empty() && llama_state_seq_get_data_ext(ctx, b.data(), b.size(), 0, flags) == b.size(), "checkpoint save failed");
    return b;
}

std::vector<float> replay(llama_context *ctx, Tokens &suffix, int vocab) {
    expect(suffix.size() == n_steps, "fixture needs eight fixed continuation tokens");
    std::vector<float> logits;
    logits.reserve(suffix.size() * vocab);
    for (size_t i = 0; i < suffix.size(); ++i) {
        decode(ctx, suffix, i, 1);
        const auto *l = llama_get_logits_ith(ctx, -1);
        expect(l && std::all_of(l, l + vocab, [](float x) { return std::isfinite(x); }), "invalid logits");
        logits.insert(logits.end(), l, l + vocab);
    }
    return logits;
}

template<typename F> F proc(ggml_backend_reg_t reg, const char *name) {
    void *ptr = ggml_backend_reg_get_proc_address(reg, name);
    F fn = nullptr;
    static_assert(sizeof(fn) == sizeof(ptr));
    std::memcpy(&fn, &ptr, sizeof(fn));
    expect(fn, name);
    return fn;
}

struct Upload {
    explicit Upload(ggml_backend_dev_t dev) :
        backend(ggml_backend_dev_init(dev, nullptr), ggml_backend_free),
        copy(proc<decltype(copy)>(ggml_backend_dev_backend_reg(dev), "ggml_backend_vk_r10_copy")),
        submit(proc<decltype(submit)>(ggml_backend_dev_backend_reg(dev), "ggml_backend_vk_r10_submit")),
        wait(proc<decltype(wait)>(ggml_backend_dev_backend_reg(dev), "ggml_backend_vk_r10_wait")),
        bind(proc<decltype(bind)>(ggml_backend_dev_backend_reg(dev), "ggml_backend_vk_r10_bind")) {
        expect(bool(backend), "upload backend missing");
    }
    Backend backend;
    bool (*copy)(ggml_backend_t, ggml_tensor *, const void *, size_t, size_t);
    uint64_t (*submit)(ggml_backend_t);
    int (*wait)(ggml_backend_t, uint64_t, uint64_t);
    bool (*bind)(ggml_backend_t, uint64_t);
};

// Optional queue witness for the real compute-completion snapshot. All model
// and backend API calls continue to originate from the native worker thread.
class ComputeWitness {
public:
    explicit ComputeWitness(const std::filesystem::path &p) : dll_(LoadLibraryW(std::filesystem::absolute(p).c_str())) {
        expect(dll_, "could not load native compute witness");
        const auto arm = GetProcAddress(dll_, "r10_trace_arm"), wait = GetProcAddress(dll_, "r10_trace_wait");
        static_assert(sizeof(arm) == sizeof(arm_) && sizeof(wait) == sizeof(wait_));
        std::memcpy(&arm_, &arm, sizeof(arm_));
        std::memcpy(&wait_, &wait, sizeof(wait_));
        expect(arm_ && wait_, "compute witness ABI not available");
    }
    ~ComputeWitness() { FreeLibrary(dll_); }
    ComputeWitness(const ComputeWitness &) = delete;
    ComputeWitness &operator=(const ComputeWitness &) = delete;
    void record(ggml_backend_t backend, ggml_backend_event_t event) {
        expect(arm_(++id_) > 0, "could not arm compute witness");
        ggml_backend_event_record(event, backend);
        arm_(0);
    }
    R10TraceRecord poll(int &ready) const {
        R10TraceRecord evidence{};
        const int status = wait_(id_, 0, &evidence);
        expect(evidence.queue && evidence.timeline_value &&
               (status == R10_TRACE_OK || status == R10_TRACE_WAIT_TIMEOUT), "could not poll real compute fence");
        ready = status == R10_TRACE_OK;
        return evidence;
    }
private:
    HMODULE dll_;
    R10TraceArm arm_{};
    R10TraceWait wait_{};
    uint64_t id_ = 0;
};

struct Disk {
    explicit Disk(std::filesystem::path p) : path(std::move(p)) {
        expect(!std::filesystem::exists(path), "need a new SSD fixture path");
    }
    ~Disk() { std::error_code ignore; std::filesystem::remove(path, ignore); }
    void stage(const Bytes &b) {
        const Bytes pad(round_to(b.size(), 4096) - b.size());
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(b.data()), b.size());
        if (!pad.empty()) out.write(reinterpret_cast<const char *>(pad.data()), pad.size());
        out.close();
        expect(bool(out), "could not write unbuffered-read fixture");
    }
    size_t read(void *target, size_t logical) const {
        const size_t bytes = round_to(logical, 4096);
        HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                 FILE_FLAG_NO_BUFFERING, nullptr);
        expect(file != INVALID_HANDLE_VALUE, "could not open SSD object");
        DWORD got = 0;
        const BOOL ok = ReadFile(file, target, DWORD(bytes), &got, nullptr);
        CloseHandle(file);
        expect(ok && got == bytes, "SSD object was not read completely");
        return got;
    }
    std::filesystem::path path;
};

struct Aligned {
    explicit Aligned(size_t n) : data(VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)) {
        expect(data, "could not allocate aligned SSD read area");
    }
    ~Aligned() { VirtualFree(data, 0, MEM_RELEASE); }
    void *data;
};

struct Region { size_t source, count; std::string tensor; bool attention; };
struct Checkpoint {
    int depth = 0;
    Bytes data, skeleton, final;
    std::vector<Region> regions;
    std::vector<float> expected;
};

size_t prepare(llama_context *ctx, const Bytes &data, int depth, Shape &spans) {
    size_t count = 0;
    expect(llama_r10_prepare_seq_restore(ctx, data.data(), data.size(), 0, spans.data(), spans.size(), &count) == data.size() &&
           count && llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == depth - 1, "exact state planning failed");
    return count;
}

Checkpoint source(llama_context *ctx, Tokens &base, Tokens &suffix, int depth, int vocab) {
    clear(ctx, true);
    for (int i = 0; i < depth; i += 256) decode(ctx, base, i, 256);
    auto data = save(ctx);
    const auto recurrent = save(ctx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    constexpr size_t header = sizeof(uint32_t) + sizeof(llama_seq_id);
    expect(recurrent.size() > header && data.size() > recurrent.size() &&
           std::equal(recurrent.begin(), recurrent.begin() + header, data.begin()) &&
           std::equal(recurrent.rbegin(), recurrent.rend() - header, data.rbegin()), "unexpected checkpoint encoding");
    const size_t split = data.size() - recurrent.size() + header;
    Checkpoint result;
    result.depth = depth;
    result.data = std::move(data);
    result.skeleton = result.data;
    result.expected = replay(ctx, suffix, vocab);
    result.final = save(ctx);
    clear(ctx, true);
    Shape spans{};
    const auto count = prepare(ctx, result.data, depth, spans);
    for (size_t i = 0; i < count; ++i) {
        const auto &s = spans[i];
        const auto at = size_t(s.src - result.data.data());
        expect(s.tensor && s.tensor->buffer && s.offset % 4 == 0 && s.size && s.size % 4 == 0 &&
               at <= result.data.size() && s.size <= result.data.size() - at &&
               (at >= split || s.size <= split - at), "invalid source range in restore plan");
        result.regions.push_back({at, s.size, ggml_get_name(s.tensor), at < split});
        std::fill_n(result.skeleton.begin() + at, s.size, uint8_t(0));
    }
    clear(ctx, false);
    return result;
}

struct Slice { size_t object, offset, count; };
struct Layout {
    std::array<Bytes, 4> objects;
    std::vector<std::vector<Slice>> a, b;
};

Layout partition(const Checkpoint &a, const Checkpoint &b) {
    Layout out;
    expect(a.regions.size() == b.regions.size(), "different number of dependency spans");
    const auto append = [&](size_t obj, const uint8_t *bytes, size_t n) {
        const auto start = out.objects[obj].size();
        out.objects[obj].insert(out.objects[obj].end(), bytes, bytes + n);
        return Slice{obj, start, n};
    };
    for (size_t i = 0; i < a.regions.size(); ++i) {
        const auto &x = a.regions[i], &y = b.regions[i];
        const auto *sa = a.data.data() + x.source, *sb = b.data.data() + y.source;
        expect(x.tensor == y.tensor && x.attention == y.attention, "dependency names or roles differ by depth");
        if (x.attention) {
            expect(y.count == 2 * x.count && std::equal(sa, sa + x.count, sb), "the 512-token prefix is not byte-identical across depths");
            const auto first = append(kv0, sa, x.count);
            out.a.push_back({first});
            out.b.push_back({first, append(kv1, sb + x.count, x.count)});
        } else {
            expect(x.count == y.count, "checkpoint state shape changed across fixture depths");
            out.a.push_back({append(r512, sa, x.count)});
            out.b.push_back({append(r1024, sb, y.count)});
        }
    }
    expect(!out.objects[kv0].empty() && !out.objects[r512].empty() && out.objects[r512] != out.objects[r1024],
           "fixture should have a shared KV block and two distinct boundary checkpoints");
    return out;
}

struct EventLog {
    int rep;
    const char *arm;
    size_t seq = 0;
    Clock::time_point start = Clock::now();
    void emit(const R10StagePool &pool, const char *kind, int request = -1, int object = -1,
              size_t bytes = 0, uint64_t fence = 0, int status = -1) {
        const size_t refs = object >= 0 ? pool.get(size_t(object)).refs : 0;
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
        std::cout << rep << ',' << arm << ',' << seq++ << ',' << kind << ',' << request << ',' << object << ','
                  << pool.capacity() << ',' << pool.charged() << ',' << refs << ',' << bytes << ',' << fence << ','
                  << status << ',' << us << '\n';
        std::cout.flush();
    }
};

// Both endpoints are checked I8 descriptions of slices in existing Vulkan
// buffers. ggml passes these aliases to its native same-device VkBuffer copy;
// no application-side payload is read or written during scatter.
ggml_tensor *slice(ggml_context *meta, ggml_backend_buffer_t buffer, uintptr_t address, size_t bytes) {
    const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    const auto capacity = ggml_backend_buffer_get_size(buffer);
    expect(!ggml_backend_buffer_is_host(buffer) && address >= base && address - base <= capacity &&
           bytes && bytes <= capacity - (address - base) && (address | bytes) % 4 == 0,
           "invalid device view in scatter");
    auto *view = ggml_new_tensor_1d(meta, GGML_TYPE_I8, int64_t(bytes));
    expect(view && ggml_backend_tensor_alloc(buffer, view, reinterpret_cast<void *>(address)) == GGML_STATUS_SUCCESS,
           "could not describe device slice");
    return view;
}

class Fixture {
public:
    Fixture(ggml_backend_dev_t dev, Upload &io, llama_context *consumer, llama_context *pred, int vocab,
            size_t budget, const std::vector<size_t> &sizes, const Layout &inputs,
            const Disk &shallow_disk, const Disk &deep_disk, EventLog &log) :
        pool(budget, sizes), io_(io), dest_(consumer), pred_(pred), vocab_(vocab), inputs_(inputs),
        r0_(shallow_disk), r1_(deep_disk), log_(log),
        scatter_(ggml_backend_dev_init(dev, nullptr), ggml_backend_free),
        slab_(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), budget), ggml_backend_buffer_free),
        device_meta_(ggml_init({ggml_tensor_overhead(), nullptr, true}), ggml_free),
        max_size_(*std::max_element(sizes.begin(), sizes.end())),
        host_(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(dev), round_to(max_size_, 4096)),
              ggml_backend_buffer_free), bounce_(round_to(max_size_, 4096)) {
        expect(scatter_ && slab_ && host_ && device_meta_ && ggml_backend_buffer_get_size(slab_.get()) == budget &&
               ggml_backend_buffer_get_type(host_.get()) == ggml_backend_dev_host_buffer_type(dev) &&
               !ggml_backend_buffer_is_host(slab_.get()) && ggml_backend_buffer_get_type(slab_.get()) == ggml_backend_dev_buffer_type(dev),
               "device arena or pinned host allocation incorrect");
        base_ = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(slab_.get()));
        stage_ = slice(device_meta_.get(), slab_.get(), base_, budget);
        pinned_ = static_cast<uint8_t *>(ggml_backend_buffer_get_base(host_.get()));
        expect(pinned_, "no address for pinned staging area");
        log_.emit(pool, "device-arena", -1, -1, ggml_backend_buffer_get_size(slab_.get()));
    }

    bool offer(int req, const std::vector<size_t> &deps) {
        const bool ok = pool.reserve(req, deps);
        log_.emit(pool, ok ? "reserve" : "reject", req, int(kv0));
        return ok;
    }
    void begin_predecessor(Tokens &tokens) {
        clear(pred_, true);
        decode(pred_, tokens, 0, tokens.size());
        log_.emit(pool, "pred-submit");
    }

    void mark_predecessor(ComputeWitness &witness, ggml_backend_event_t event) {
        witness.record(scatter_.get(), event);
    }

    void prior_done() {
        llama_synchronize(pred_);
        log_.emit(pool, "pred-end");
    }

    void stage_host(size_t object, int req = -1) {
        const auto bytes = pool.get(object).bytes;
        if (object == r512 || object == r1024) {
            const auto &disk = object == r512 ? r0_ : r1_;
            const size_t got = disk.read(bounce_.data, bytes);
            log_.emit(pool, "disk-read", req, int(object), got);
            std::memcpy(pinned_, bounce_.data, bytes);
        } else {
            std::memcpy(pinned_, inputs_.objects[object].data(), bytes);
        }
    }

    uint64_t issue(size_t object) {
        expect(pool.get(object).state == R10StagePool::State::reserved, "only reserved objects can be issued");
        const auto bytes = pool.get(object).bytes;
        stage_host(object);
        pool.pieces(object, 0, bytes, [&](size_t dev, size_t off, size_t chunk) {
            expect(io_.copy(io_.backend.get(), stage_, pinned_ + off, dev, chunk), "native stage upload rejected");
        });
        const auto fence = io_.submit(io_.backend.get());
        expect(fence, "missing native upload completion signal");
        pool.issued(object, fence);
        log_.emit(pool, "upload-submit", -1, int(object), bytes, fence);
        return fence;
    }

    int poll(uint64_t fence) { return io_.wait(io_.backend.get(), fence, 0); }

    void finish(size_t object, uint64_t fence, bool already_observed = false) {
        expect(already_observed || io_.wait(io_.backend.get(), fence, 5'000'000'000ULL) == 1, "upload did not complete");
        expect(io_.bind(io_.backend.get(), fence), "upload was not bound to compute queue");
        pool.landed(object, fence);
        log_.emit(pool, "upload-ready", -1, int(object), pool.get(object).bytes, fence);
    }

    void ensure(const std::vector<size_t> &deps) {
        for (auto id : deps) if (pool.get(id).state != R10StagePool::State::ready) finish(id, issue(id));
    }

    void consume(int req, const Checkpoint &ref, Tokens &suffix, const std::vector<size_t> &deps,
                 const std::vector<std::vector<Slice>> &layout, bool reuse_active_kv = false,
                 Tokens *full_prefix = nullptr, const Checkpoint *final_ref = nullptr) {
        restore_staged(req, ref, deps, layout, reuse_active_kv);
        if (full_prefix) fill_prefix(req, *full_prefix, ref.depth);
        verify(req, suffix, final_ref ? *final_ref : ref);
    }

    void restore_staged(int req, const Checkpoint &ref, const std::vector<size_t> &deps,
                        const std::vector<std::vector<Slice>> &layout, bool reuse_active_kv) {
        expect(pool.matches(req, deps) && pool.consumable(req), "consume without exact ready dependencies");
        expect(!borrowed_kv_ || reuse_active_kv, "borrowed page has no staged payload to copy");
        phase = "device-consume";
        clear(dest_, !reuse_active_kv);
        Shape spans{};
        const auto count = prepare(dest_, ref.skeleton, ref.depth, spans);
        expect(count == layout.size(), "target tensor layout changed");
        // No payload in the host image; the real cached object is the sole
        // source for every native device scatter below.
        const Meta aliases(ggml_init({2 * (count + 10 + pool.capacity() / R10StagePool::page) * ggml_tensor_overhead(),
                                     nullptr, true}), ggml_free);
        expect(bool(aliases), "scatter descriptors could not be allocated");
        const auto scatter_begin = Clock::now();
        size_t copied = 0, reused = 0;
        if (!req) active_kv_.clear();
        for (size_t i = 0; i < count; ++i) {
            const auto &s = spans[i];
            expect(s.size == ref.regions.at(i).count && ggml_get_name(s.tensor) == ref.regions.at(i).tensor &&
                   ggml_backend_buffer_get_type(s.tensor->buffer) == ggml_backend_buffer_get_type(slab_.get()),
                   "target tensor role, size or allocation type changed");
            size_t into_tensor = 0;
            for (const auto &x : layout[i]) {
                const auto dst_address = reinterpret_cast<uintptr_t>(s.tensor->data) + s.offset + into_tensor;
                if (x.object == kv0) {
                    const RegionOnDevice region{s.tensor->buffer, dst_address, x.count};
                    if (!req) active_kv_.emplace(x.offset, region);
                    if (reuse_active_kv) {
                        const auto old = active_kv_.find(x.offset);
                        expect(old != active_kv_.end() && old->second.buffer == region.buffer &&
                               old->second.base == region.base && old->second.bytes == region.bytes,
                               "native KV cell mapping changed; active prefix reuse is invalid");
                        reused += x.count;
                        into_tensor += x.count;
                        continue;
                    }
                }
                pool.pieces(x.object, x.offset, x.count, [&](size_t arena, size_t written, size_t chunk) {
                    auto *src = slice(aliases.get(), slab_.get(), base_ + arena, chunk);
                    auto *dst = slice(aliases.get(), s.tensor->buffer, dst_address + written, chunk);
                    ggml_backend_tensor_copy_async(scatter_.get(), scatter_.get(), src, dst);
                    copied += chunk;
                });
                into_tensor += x.count;
            }
            expect(into_tensor == s.size, "scatter does not cover the full destination");
        }
        ggml_backend_synchronize(scatter_.get());
        const int scatter_us = int(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - scatter_begin).count());
        if (reuse_active_kv) {
            expect(reused == pool.get(kv0).bytes, "active reuse must cover the full identity-matched KV block");
            log_.emit(pool, "active-kv-reused", req, int(kv0), reused);
        }
        log_.emit(pool, "device-copy", req, -1, copied, 0, scatter_us);
        release(req);
    }

    void borrow_prefix(int req, const std::vector<size_t> &deps) {
        expect(pool.matches(req, deps) && !active_kv_.empty() && !borrowed_kv_ &&
               std::find(deps.begin(), deps.end(), kv0) != deps.end(), "invalid native active-KV lease");
        pool.borrow_active(kv0);
        borrowed_kv_ = true;
        log_.emit(pool, "active-kv-borrow", req, int(kv0), pool.get(kv0).bytes);
    }

    void begin_suffix(Tokens &suffix, ComputeWitness &witness, ggml_backend_event_t event) {
        decode(dest_, suffix, 0, 1);
        witness.record(scatter_.get(), event);
        log_.emit(pool, "first-suffix-submitted", 0);
    }

    void complete_suffix(int req, Tokens &suffix, const Checkpoint &ref) {
        expect(suffix.size() == n_steps, "wrong fixture suffix length");
        std::vector<float> result;
        for (size_t i = 0; i < suffix.size(); ++i) {
            if (i) decode(dest_, suffix, i, 1);
            const auto *l = llama_get_logits_ith(dest_, -1);
            expect(l, "first-suffix logits unavailable");
            result.insert(result.end(), l, l + vocab_);
        }
        const bool equal = result == ref.expected && save(dest_) == ref.final;
        log_.emit(pool, "consumed", req, -1, n_steps, 0, int(equal));
        expect(equal, "overlapped first-suffix logits or final state differs from reference");
    }

    void recompute(int req, Tokens &base, Tokens &suffix, const Checkpoint &ref) {
        clear(dest_, true);
        fill_prefix(req, base, 0);
        verify(req, suffix, ref);
    }

    // Idle-head cold restore when the prefetch budget cannot hold even one
    // candidate. Write directly into the actual active sequence buffer using
    // its already allocated execution budget. No device arena pages or second
    // model context are used as an overflow pool. This entry requires an idle
    // transfer lane and no already resident/shared objects.
    void demand_cold(int req, const Checkpoint &ref, const Layout &source_layout, Tokens &base, Tokens &suffix,
                     const Checkpoint &final_ref) {
        const auto &deps = ref.depth == 512 ? shallow : deep;
        prepare_cold(req, ref, source_layout);
        for (auto id : deps) finish_cold(id, issue_cold(id));
        take_cold(req, deps);
        fill_prefix(req, base, ref.depth);
        verify(req, suffix, final_ref);
    }

    void prepare_cold(int req, const Checkpoint &ref, const Layout &source_layout) {
        phase = "direct-head-demand";
        expect(cold_req_ < 0 && !pool.charged() && !borrowed_kv_, "cold demand cannot bypass a leased pool");
        clear(dest_, true);
        cold_ref_ = &ref;
        cold_layout_ = ref.depth == 512 ? &source_layout.a : &source_layout.b;
        cold_req_ = req;
        cold_count_ = prepare(dest_, ref.skeleton, ref.depth, cold_spans_);
        expect(cold_count_ == cold_layout_->size(), "wrong number of cold-demand spans");
        if (!req) active_kv_.clear();
        for (size_t i = 0; i < cold_count_; ++i) {
            const auto &s = cold_spans_[i];
            expect(s.size == ref.regions.at(i).count && ggml_get_name(s.tensor) == ref.regions.at(i).tensor &&
                   ggml_backend_buffer_get_type(s.tensor->buffer) == ggml_backend_buffer_get_type(slab_.get()),
                   "cold-demand tensor role, size or allocation type changed");
            size_t offset = 0;
            for (const auto &x : cold_layout_->at(i)) {
                if (!req && x.object == kv0) active_kv_.emplace(x.offset, RegionOnDevice{
                    s.tensor->buffer, reinterpret_cast<uintptr_t>(s.tensor->data) + s.offset + offset, x.count});
                offset += x.count;
            }
            expect(offset == s.size, "cold demand does not cover active tensor");
        }
        log_.emit(pool, "direct-head-demand", req, -1, size_t(ref.depth));
    }

    uint64_t issue_cold(size_t id) {
        expect(cold_req_ >= 0 && !cold_fence_ && !cold_ready_.count(id), "invalid cold transfer transition");
        stage_host(id, cold_req_);
        size_t sent = 0;
        for (size_t i = 0; i < cold_count_; ++i) {
            const auto &s = cold_spans_[i];
            size_t offset = 0;
            for (const auto &x : cold_layout_->at(i)) {
                if (x.object == id) {
                    expect(x.offset == sent && x.count <= pool.get(id).bytes - sent &&
                           io_.copy(io_.backend.get(), s.tensor, pinned_ + x.offset, s.offset + offset, x.count),
                           "native direct demand upload rejected");
                    sent += x.count;
                }
                offset += x.count;
            }
        }
        expect(sent && sent == pool.get(id).bytes, "cold demand did not upload the complete object");
        cold_object_ = id;
        cold_fence_ = io_.submit(io_.backend.get());
        expect(cold_fence_, "missing cold-demand signal");
        log_.emit(pool, "direct-upload-submit", cold_req_, int(id), sent, cold_fence_);
        return cold_fence_;
    }

    void finish_cold(size_t id, uint64_t ticket, bool observed = false) {
        expect(cold_req_ >= 0 && ticket && ticket == cold_fence_ && cold_object_ == id,
               "cold demand completion identity changed");
        expect((observed || io_.wait(io_.backend.get(), ticket, 5'000'000'000ULL) == 1) && io_.bind(io_.backend.get(), ticket),
               "direct head demand was not visible to execution");
        cold_fence_ = 0;
        cold_ready_.insert(id);
        log_.emit(pool, "direct-upload-ready", cold_req_, int(id), pool.get(id).bytes, ticket);
    }

    void take_cold(int req, const std::vector<size_t> &deps) {
        expect(cold_req_ == req && !cold_fence_ && cold_ready_ == std::set<size_t>(deps.begin(), deps.end()),
               "incomplete direct head demand");
        cold_ref_ = nullptr;
        cold_layout_ = nullptr;
        cold_req_ = -1;
        cold_ready_.clear();
    }

    bool is_cold(int req, int depth) const { return cold_req_ == req && cold_ref_ && cold_ref_->depth == depth; }
    bool has_borrowed_prefix() const { return borrowed_kv_; }
    llama_context *compute_context() const { return dest_; }

    void fill_prefix(int req, Tokens &base, size_t depth) {
        expect(depth <= base.size() && (base.size() - depth) % 256 == 0, "bad diagnostic prefix split");
        log_.emit(pool, "prefix-work", req, -1, base.size() - depth);
        for (size_t i = depth; i < base.size(); i += 256) decode(dest_, base, i, 256);
    }

    void verify(int req, Tokens &suffix, const Checkpoint &ref) {
        const auto actual = replay(dest_, suffix, vocab_);
        const bool equal = actual == ref.expected && save(dest_) == ref.final;
        log_.emit(pool, "consumed", req, -1, n_steps, 0, int(equal));
        expect(equal, "model did not consume correct staged data");
    }

    void release(int req) {
        pool.release(req);
        if (!pool.get(kv0).refs) borrowed_kv_ = false;
        log_.emit(pool, "release", req, int(kv0));
    }

    R10StagePool pool;
private:
    struct RegionOnDevice { ggml_backend_buffer_t buffer; uintptr_t base; size_t bytes; };
    Upload &io_;
    llama_context *dest_, *pred_;
    int vocab_;
    const Layout &inputs_;
    const Disk &r0_, &r1_;
    EventLog &log_;
    Backend scatter_;
    Buffer slab_;
    Meta device_meta_;
    const size_t max_size_;
    Buffer host_;
    Aligned bounce_;
    uintptr_t base_ = 0;
    ggml_tensor *stage_ = nullptr;
    uint8_t *pinned_ = nullptr;
    std::map<size_t, RegionOnDevice> active_kv_;
    bool borrowed_kv_ = false;
    Shape cold_spans_{};
    const Checkpoint *cold_ref_ = nullptr;
    const std::vector<std::vector<Slice>> *cold_layout_ = nullptr;
    std::set<size_t> cold_ready_;
    size_t cold_count_ = 0, cold_object_ = 0;
    int cold_req_ = -1;
    uint64_t cold_fence_ = 0;
};

// Short online diagnostic: A's boundary is fixed and B is a real action from
// the controller. No claim of a general executor or representative samples.
void online_case(Fixture &exp, EventLog &log, const Checkpoint &a, const Checkpoint &b,
                 Tokens &base, Tokens &suffix_a, Tokens &suffix_b, Tokens &preceding, const Layout &layout,
                 ComputeWitness *witness, ggml_backend_dev_t dev, bool after) {
    for (size_t i = 0; i < layout.objects.size(); ++i)
        log.emit(exp.pool, "catalog-size", -1, int(i), layout.objects[i].size());
    expect(exp.offer(0, shallow), "shallow reservation failed");
    exp.begin_predecessor(preceding);
    const std::unique_ptr<ggml_backend_event, decltype(&ggml_backend_event_free)> marker(
        witness ? ggml_backend_event_new(dev) : nullptr, ggml_backend_event_free);
    if (witness) {
        expect(bool(marker), "could not allocate real compute marker");
        exp.mark_predecessor(*witness, marker.get());
    }
    if (after) {
        expect(bool(marker), "completed-compute case must use the independent witness");
        ggml_backend_event_synchronize(marker.get());
    }
    const auto ticket = exp.issue(kv0);
    phase = "online-boundary";
    const auto status = exp.poll(ticket);
    expect(status >= 0, "native transfer poll failed");
    if (status == 1) exp.finish(kv0, ticket, true);
    if (witness) {
        int ready = -1;
        const auto evidence = witness->poll(ready);
        expect(!after || ready == 1, "counterfactual completed predecessor was not signalled");
        log.emit(exp.pool, "pred-query", -1, -1, 0, evidence.timeline_value, ready);
    }
    // Request B arrives at this native event. Both recorded statuses are
    // zero-timeout device queries when a compute witness is enabled.
    log.emit(exp.pool, "online-decision", 1, int(kv0), 0, ticket, status);
    std::string reply;
    expect(bool(std::getline(std::cin, reply)) && (reply == "0" || reply == "512" || reply == "1024"),
           "missing or illegal online boundary");
    const int chosen = std::stoi(reply);
    log.emit(exp.pool, "online-issued", 1, -1, size_t(chosen));
    const auto &deps = chosen == 512 ? shallow : deep;
    const bool prefetched = chosen && exp.offer(1, deps);
    if (status == 0) exp.finish(kv0, ticket);
    exp.ensure(shallow);
    exp.prior_done();
    exp.consume(0, a, suffix_a, shallow, layout.a);
    if (!chosen) exp.recompute(1, base, suffix_b, b);
    else {
        if (!prefetched) {
            expect(exp.offer(1, deps), "online demand did not fit diagnostic device arena");
            log.emit(exp.pool, "head-demand", 1);
        }
        exp.ensure(deps);
        exp.consume(1, chosen == 512 ? a : b, suffix_b, deps, chosen == 512 ? layout.a : layout.b, false, &base, &b);
    }
    expect(exp.pool.charged() == 0, "online device arena pages not released");
    log.emit(exp.pool, "case-end");
}

void head_demand_case(Fixture &exp, EventLog &log, int chosen, bool cold, const Checkpoint &a, const Checkpoint &b,
                      Tokens &base, Tokens &suffix_b, const Layout &layout) {
    const auto &deps = chosen == 512 ? shallow : deep;
    for (size_t i = 0; i < layout.objects.size(); ++i)
        log.emit(exp.pool, "catalog-size", -1, int(i), layout.objects[i].size());
    expect(exp.offer(1, deps) != cold, "prefetch must fail exactly when cold candidate exceeds stage capacity");
    const auto &ref = chosen == 512 ? a : b;
    if (cold) exp.demand_cold(1, ref, layout, base, suffix_b, b);
    else {
        exp.ensure(deps);
        exp.consume(1, ref, suffix_b, deps, chosen == 512 ? layout.a : layout.b, false, &base, &b);
    }
    log.emit(exp.pool, "case-end");
}

void native_borrow_case(Fixture &exp, EventLog &log, bool borrow, const Checkpoint &a, const Checkpoint &b,
                        Tokens &suffix_a, Tokens &suffix_b, const Layout &layout,
                        ComputeWitness &witness, ggml_backend_dev_t dev) {
    for (size_t i = 0; i < layout.objects.size(); ++i)
        log.emit(exp.pool, "catalog-size", -1, int(i), layout.objects[i].size());
    expect(exp.offer(0, shallow), "active-KV donor could not fit in prefetch pool");
    exp.ensure(shallow);
    exp.restore_staged(0, a, shallow, layout.a, false);
    expect(exp.pool.charged() == 0, "donor left behind prefetched KV");
    const std::unique_ptr<ggml_backend_event, decltype(&ggml_backend_event_free)> event(
        ggml_backend_event_new(dev), ggml_backend_event_free);
    expect(bool(event), "could not allocate suffix witness");
    exp.begin_suffix(suffix_a, witness, event.get());
    int ready = -1;
    const auto evidence = witness.poll(ready);
    log.emit(exp.pool, "borrow-donor-query", 0, -1, 0, evidence.timeline_value, ready);
    expect(exp.offer(1, deep), "could not admit successor dependencies");
    if (borrow) exp.borrow_prefix(1, deep);
    exp.ensure(deep);
    exp.complete_suffix(0, suffix_a, a);
    exp.consume(1, b, suffix_b, deep, layout.b, borrow);
    expect(exp.pool.charged() == 0, "native borrowed capacity was not released");
    log.emit(exp.pool, "case-end");
}

// Physical backend for the reusable LiveFIFO controller. This adapter knows
// the companion three-request tensor catalogue. All Vulkan calls stay on this
// thread; hardware signals, not predicted durations, complete controller work.
void fifo_link(Fixture &exp, EventLog &log, Tokens &preceding, Tokens &base, Tokens &sa, Tokens &sb,
               const Checkpoint &a, const Checkpoint &b, const Layout &layout, ComputeWitness &witness,
               ggml_backend_dev_t dev, int vocab) {
    auto *ctx = exp.compute_context();
    clear(ctx, true);
    decode(ctx, preceding, 0, preceding.size());
    const auto preceding_ref = save(ctx);
    Tokens short_base(base.begin(), base.begin() + 512);
    const std::unique_ptr<ggml_backend_event, decltype(&ggml_backend_event_free)> marker(
        ggml_backend_event_new(dev), ggml_backend_event_free);
    expect(bool(marker), "FIFO worker could not allocate compute marker");
    for (size_t i = 0; i < layout.objects.size(); ++i)
        log.emit(exp.pool, "catalog-size", -1, int(i), layout.objects[i].size());
    std::cout << "!open " << log.rep << ' ' << exp.pool.capacity() << std::endl;
    int active = -1, next_compute = 0;
    size_t pos = 0, xfer_id = 0;
    uint64_t fence = 0;
    bool direct = false;
    for (std::string line; std::getline(std::cin, line);) {
        std::istringstream in(line);
        std::string command;
        in >> command;
        if (command == "reserve") {
            int req = -1, depth = -1, borrow = -1;
            expect(bool(in >> req >> depth >> borrow) && (req == 1 || req == 2) &&
                   (depth == 512 || (req == 2 && depth == 1024)) && (borrow == 0 || (borrow == 1 && active == 1 && req == 2)),
                   "bad native reserve command");
            const auto &deps = depth == 512 ? shallow : deep;
            expect(exp.offer(req - 1, deps), "native arena differs from controller union accounting");
            if (borrow) exp.borrow_prefix(req - 1, deps);
            std::cout << "!ok" << std::endl;
        } else if (command == "read") {
            int req = -1, depth = -1;
            std::string lane;
            expect(bool(in >> req >> depth >> xfer_id >> lane) && !fence && (req == 1 || req == 2) &&
                   (depth == 512 || (req == 2 && depth == 1024)) && (lane == "demand" || lane == "prefetch"),
                   "bad native read command");
            const auto &deps = depth == 512 ? shallow : deep;
            expect(std::find(deps.begin(), deps.end(), xfer_id) != deps.end(), "native read not in selected boundary");
            direct = exp.pool.get(xfer_id).state == R10StagePool::State::absent;
            if (direct) {
                expect(lane == "demand" && active < 0 && next_compute == req, "cold direct read not owned by idle FIFO head");
                if (!exp.is_cold(req - 1, depth)) exp.prepare_cold(req - 1, depth == 512 ? a : b, layout);
                fence = exp.issue_cold(xfer_id);
            } else fence = exp.issue(xfer_id);
            std::cout << "!ok " << fence << std::endl;
        } else if (command == "compute") {
            int req = -1, depth = -1, released = -1, shared = -1;
            expect(bool(in >> req >> depth >> released >> shared) && active < 0 && req == next_compute && req <= 2 &&
                   (depth == 0 || ((req == 1 || req == 2) && depth == 512) || (req == 2 && depth == 1024)),
                   "bad native compute command");
            active = req;
            pos = 0;
            if (!req) {
                clear(ctx, true);
                decode(ctx, preceding, 0, preceding.size());
            } else {
                if (!depth) {
                    expect(!exp.has_borrowed_prefix(), "recompute would overwrite a leased native KV object");
                    clear(ctx, true);
                } else {
                    const auto &deps = depth == 512 ? shallow : deep;
                    const auto &ref = depth == 512 ? a : b;
                    if (exp.is_cold(req - 1, depth)) {
                        expect(!released && !shared, "native direct leg unexpectedly has staged owners");
                        exp.take_cold(req - 1, deps);
                    } else {
                        expect(released, "native staged leg has no compute owner");
                        exp.restore_staged(req - 1, ref, deps, depth == 512 ? layout.a : layout.b, exp.has_borrowed_prefix());
                    }
                }
                exp.fill_prefix(req - 1, req == 1 ? short_base : base, size_t(depth));
                decode(ctx, req == 1 ? sa : sb, 0, 1);
            }
            exp.mark_predecessor(witness, marker.get());
            log.emit(exp.pool, "fifo-compute-submit", active, -1, size_t(depth));
            std::cout << "!ok" << std::endl;
        } else if (command == "poll") {
            uint64_t landed = 0;
            if (fence) {
                const int ready = exp.poll(fence);
                expect(ready >= 0, "FIFO transfer poll failed");
                if (ready) {
                    if (direct) exp.finish_cold(xfer_id, fence, true);
                    else exp.finish(xfer_id, fence, true);
                    landed = fence;
                    fence = 0;
                }
            }
            int done = -1;
            if (active >= 0) {
                int ready = 0;
                const auto signal = witness.poll(ready);
                if (ready) {
                    if (active) {
                        const auto &ref = active == 1 ? a : b;
                        const auto *l = llama_get_logits_ith(ctx, -1);
                        expect(l && std::equal(l, l + vocab, ref.expected.begin() + pos * vocab), "FIFO logits differ from reference");
                        if (++pos < n_steps) {
                            decode(ctx, active == 1 ? sa : sb, pos, 1);
                            exp.mark_predecessor(witness, marker.get());
                        } else expect(save(ctx) == ref.final, "FIFO native final state differs");
                    } else expect(save(ctx) == preceding_ref, "FIFO full recompute differs");
                    if (!active || pos == n_steps) {
                        log.emit(exp.pool, "fifo-compute-done", active, -1, pos, signal.timeline_value, 1);
                        done = active;
                        active = -1;
                        ++next_compute;
                    }
                }
            }
            std::cout << "!poll " << landed << ' ' << done << std::endl;
        } else if (command == "finish") {
            expect(next_compute == 3 && active < 0 && !fence && !exp.pool.charged(), "native FIFO ended with pending work");
            log.emit(exp.pool, "case-end");
            std::cout << "!done" << std::endl;
            return;
        } else expect(false, "unknown native FIFO protocol command");
    }
    expect(false, "native FIFO pipe closed before completion");
}
}

int main(int argc, char **argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    SetUnhandledExceptionFilter(fault);
    const std::string mode = argc >= 7 ? argv[6] : "";
    const bool online = mode == "online" || mode == "online-observed", forced = mode == "online-forced";
    const bool observed = mode == "online-observed", demand = mode == "head-demand";
    const bool native_borrow = mode == "active-kv-borrow";
    const bool fifo = mode == "fifo-link" || mode == "fifo-link-cold";
    expect(argc == 6 || (argc == 7 && !observed && (online || forced || demand || mode == "active-kv")) ||
           (argc == 8 && (observed || native_borrow || fifo)), "usage: r10_stage_pool_probe MODEL DLLDIR DEVICE NEW_PREFIX REPS [MODE [OBSERVER_DLL]]");
    expect(std::getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") && std::getenv("GGML_VK_R10_CONCURRENT_BUFFERS"),
           "explicit async-transfer and concurrent-buffer opt-ins are required");
    const int reps = std::stoi(argv[5]);
    expect(reps > 0, "repeat count must be positive");
    Disk on_disk0(std::filesystem::absolute(std::string(argv[4]) + ".r512"));
    Disk on_disk1(std::filesystem::absolute(std::string(argv[4]) + ".r1024"));
    const auto witness = observed || native_borrow || fifo ? std::make_unique<ComputeWitness>(argv[7]) : nullptr;
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devs[]{ggml_backend_dev_by_name(argv[3]), nullptr};
    expect(devs[0] && ggml_backend_dev_host_buffer_type(devs[0]), "requested Vulkan device/pinned storage unavailable");
    Upload upload(devs[0]);
    auto mp = llama_model_default_params();
    mp.devices = devs;
    mp.n_gpu_layers = 99;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    expect(bool(model), "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    const Context consumer(llama_init_from_model(model.get(), cp), llama_free), pred(llama_init_from_model(model.get(), cp), llama_free);
    expect(consumer && pred, "could not initialize two independent model contexts");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::mt19937 prompt(29690), order(1023133);
    std::cout << "rep,arm,event_no,event,request,object,capacity_bytes,charged_bytes,refs,bytes,fence,status,wall_us\n";

    for (int rep = -1; rep < reps; ++rep) {
        phase = "reference";
        Tokens base(1024), suffix_a(n_steps), suffix_b(n_steps), preceding(512);
        for (auto *t : {&base, &suffix_a, &suffix_b, &preceding}) for (auto &id : *t) id = 1000 + prompt() % 14000;
        const auto a = source(consumer.get(), base, suffix_a, 512, vocab);
        const auto b = source(consumer.get(), base, suffix_b, 1024, vocab);
        const auto layout = partition(a, b);
        on_disk0.stage(layout.objects[r512]);
        on_disk1.stage(layout.objects[r1024]);
        std::vector<size_t> sizes;
        size_t union_bytes = 0;
        for (const auto &x : layout.objects) {
            sizes.push_back(x.size());
            union_bytes += R10StagePool::cost(x.size());
        }
        std::vector<const char *> arms = {"exact-union", "union-minus-page", "inflight-cancel"};
        if (mode == "active-kv") arms = {"exact-union", "exact-union-active-kv"};
        if (online) arms = {"entry-replay-exact", "local-cost-exact", "entry-replay-tight", "local-cost-tight"};
        if (observed) arms.insert(arms.end(), {"entry-replay-after", "local-cost-after"});
        if (forced) arms = {"forced-0", "forced-512", "forced-1024"};
        if (demand) arms = {"cold-512", "staged-512", "cold-1024", "staged-1024"};
        if (native_borrow) arms = {"active-copy-control", "active-borrow"};
        if (fifo) arms = {mode.c_str()};
        std::shuffle(arms.begin(), arms.end(), order);
        for (const char *arm : arms) {
            phase = "arena";
            const bool low = std::strcmp(arm, "union-minus-page") == 0 || std::strstr(arm, "-tight");
            const bool cancel = std::strcmp(arm, "inflight-cancel") == 0;
            const bool reuse = std::strcmp(arm, "exact-union-active-kv") == 0;
            EventLog log{rep, arm};
            const bool cold = std::strncmp(arm, "cold-", 5) == 0;
            const int demand_depth = std::strstr(arm, "-512") ? 512 : 1024;
            const auto required = demand_depth == 512 ? R10StagePool::cost(sizes[kv0]) + R10StagePool::cost(sizes[r512]) :
                R10StagePool::cost(sizes[kv0]) + R10StagePool::cost(sizes[kv1]) + R10StagePool::cost(sizes[r1024]);
            const auto n = mode == "fifo-link-cold" ? R10StagePool::cost(sizes[kv0]) + R10StagePool::cost(sizes[r512]) - R10StagePool::page :
                (demand || native_borrow) ? required - (cold ? R10StagePool::page : 0) : union_bytes - (low ? R10StagePool::page : 0);
            Fixture exp(devs[0], upload, consumer.get(), pred.get(), vocab, n, sizes, layout, on_disk0, on_disk1, log);
            if (fifo) {
                fifo_link(exp, log, preceding, base, suffix_a, suffix_b, a, b, layout, *witness, devs[0], vocab);
                continue;
            }
            if (native_borrow) {
                native_borrow_case(exp, log, std::strcmp(arm, "active-borrow") == 0, a, b, suffix_a, suffix_b, layout, *witness, devs[0]);
                continue;
            }
            if (demand) {
                head_demand_case(exp, log, demand_depth, cold, a, b, base, suffix_b, layout);
                continue;
            }
            if (online || forced) {
                online_case(exp, log, a, b, base, suffix_a, suffix_b, preceding, layout, witness.get(), devs[0], std::strstr(arm, "-after"));
                continue;
            }
            expect(exp.offer(0, shallow), "shallow reservation failed");
            exp.begin_predecessor(preceding);
            phase = "in-flight";
            const auto ticket = exp.issue(kv0);
            expect(!exp.pool.consumable(0), "in-flight object was consumable");
            log.emit(exp.pool, "early-consume-rejected", 0, int(kv0));
            const int status = exp.poll(ticket);
            expect(status >= 0, "transfer poll failed");
            log.emit(exp.pool, "handoff-at-sample", 1, int(kv0), 0, ticket, status);
            expect(exp.offer(1, deep) != low, "capacity did not separate union and union-minus-page");
            if (!low) {
                expect(exp.pool.get(kv0).refs == 2 && exp.pool.charged() == union_bytes, "shared KV was charged twice");
                expect(!exp.pool.matches(1, {kv1, r1024}), "missing shared KV accepted");
                log.emit(exp.pool, "missing-kv-rejected", 1, int(kv0));
                expect(!exp.pool.matches(1, {kv0, kv1, r512}), "wrong checkpoint accepted");
                log.emit(exp.pool, "wrong-checkpoint-rejected", 1, int(r512));
            }
            if (cancel) exp.release(0);
            exp.finish(kv0, ticket, status == 1);
            if (!cancel) exp.ensure(shallow);
            exp.prior_done();
            if (!cancel) exp.consume(0, a, suffix_a, shallow, layout.a);
            if (low) {
                expect(exp.pool.charged() == 0 && exp.offer(1, deep), "rejected prefetch did not admit demand after release");
                log.emit(exp.pool, "head-demand", 1);
            }
            exp.ensure(deep);
            exp.consume(1, b, suffix_b, deep, layout.b, reuse);
            expect(exp.pool.charged() == 0, "device arena pages not released");
            log.emit(exp.pool, "case-end");
        }
    }
    phase = "shutdown";
    return 0;
} catch (const std::exception &e) {
    std::cerr << "r10-stage: " << e.what() << '\n';
    return 1;
}
