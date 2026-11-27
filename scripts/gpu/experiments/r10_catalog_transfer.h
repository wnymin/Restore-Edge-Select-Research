#pragma once

#include "r10_catalog_source.h"
#include "r10_catalog_target.h"
#include "r10_stage_pool.h"

#include <cstring>
#include <memory>

namespace r10cat {
// Bytes travel through one host-to-device lane. A disk operation occupies that
// lane until the following upload is visible, but does not block compute polls.
class DeviceTransfer {
public:
    using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    using TensorMeta = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    struct Step {
        uint64_t submitted = 0, completed = 0;
        size_t ssd_bytes = 0;
        int64_t ssd_observation_us = 0, soft_hold_us = 0;
    };

    DeviceTransfer(ggml_backend_dev_t device, const Plan &plan, const fs::path &directory, size_t arena_bytes,
                   std::chrono::milliseconds soft_hold = std::chrono::milliseconds(0)) :
        plan_(plan), root_(directory), device_type_(ggml_backend_dev_buffer_type(device)),
        upload_(ggml_backend_dev_init(device, nullptr), ggml_backend_free),
        scatter_(ggml_backend_dev_init(device, nullptr), ggml_backend_free),
        stage_(ggml_backend_buft_alloc_buffer(device_type_, arena_bytes), ggml_backend_buffer_free),
        pinned_(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(device), maximum(plan)), ggml_backend_buffer_free),
        tensor_meta_(ggml_init({ggml_tensor_overhead(), nullptr, true}), ggml_free), disk_(maximum(plan)),
        copy_(function<decltype(copy_)>(device, "ggml_backend_vk_r10_copy")),
        submit_(function<decltype(submit_)>(device, "ggml_backend_vk_r10_submit")),
        wait_(function<decltype(wait_)>(device, "ggml_backend_vk_r10_wait")),
        bind_(function<decltype(bind_)>(device, "ggml_backend_vk_r10_bind")), soft_hold_(soft_hold) {
        need(upload_ && scatter_ && stage_ && pinned_ && tensor_meta_ && arena_bytes == ggml_backend_buffer_get_size(stage_.get()) &&
             ggml_backend_buffer_get_type(pinned_.get()) == ggml_backend_dev_host_buffer_type(device),
             "could not allocate declared native arena and pinned storage");
        arena_base_ = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(stage_.get()));
        stage_tensor_ = slice(tensor_meta_.get(), stage_.get(), arena_base_, arena_bytes);
        host_ = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pinned_.get()));
        need(host_, "native pinned allocation lacks host address");
        for (size_t id = 0; id < plan_.objects.size(); ++id) {
            const auto &obj = plan_.objects[id];
            // Read the catalogue into ordinary RAM before the timed epoch.
            if (obj.kind != 'K') continue;
            auto &blob = ram_[id];
            blob.resize(obj.bytes);
            std::ifstream f(path_for(root_, id, false), std::ios::binary);
            need(bool(f.read(reinterpret_cast<char *>(blob.data()), blob.size())) && f.peek() == std::char_traits<char>::eof(),
                 "RAM catalogue object has wrong size");
            ram_bytes_ += blob.size();
        }
    }
    ~DeviceTransfer() {
        if (ticket_) (void) wait_(upload_.get(), ticket_, UINT64_MAX);
    }
    DeviceTransfer(const DeviceTransfer &) = delete;
    DeviceTransfer &operator=(const DeviceTransfer &) = delete;

    ggml_backend_t compute_marker_backend() const { return scatter_.get(); }
    ggml_backend_buffer_type_t device_type() const { return device_type_; }
    bool busy() const { return phase_ != Phase::idle; }
    size_t resident_ram_bytes() const { return ram_bytes_; }
    size_t pinned_bytes() const { return ggml_backend_buffer_get_size(pinned_.get()); }
    size_t arena_bytes() const { return ggml_backend_buffer_get_size(stage_.get()); }

    Step read(size_t id, const Target *direct, const R10StagePool &pool) {
        need(!busy() && !ticket_ && pool.get(id).bytes == plan_.objects.at(id).bytes, "conflicting or wrong-sized transfer");
        id_ = id;
        direct_ = direct;
        const auto &o = plan_.objects[id];
        if (o.kind == 'R') {
            disk_.start(path_for(root_, id, false), o.bytes);
            phase_ = Phase::disk;
            return {};
        }
        std::memcpy(host_, ram_.at(id).data(), o.bytes);
        return issue(pool);
    }

    Step poll(const R10StagePool &pool) {
        if (phase_ == Phase::idle) return {};
        if (phase_ == Phase::disk) {
            const auto completion = disk_.poll();
            if (!completion) return {};
            if (soft_hold_.count()) {
                hold_start_ = DiskRead::Clock::now();
                phase_ = Phase::diagnostic_hold;
                return {0, 0, completion->bytes, completion->observed_us, 0};
            }
            std::memcpy(host_, disk_.payload(completion->bytes), completion->bytes);
            auto next = issue(pool);
            next.ssd_bytes = completion->bytes;
            next.ssd_observation_us = completion->observed_us;
            return next;
        }
        if (phase_ == Phase::diagnostic_hold) {
            const auto delay = DiskRead::Clock::now() - hold_start_;
            if (delay < soft_hold_) return {};
            const auto bytes = plan_.objects[id_].bytes;
            std::memcpy(host_, disk_.payload(bytes), bytes);
            auto next = issue(pool);
            next.soft_hold_us = std::chrono::duration_cast<std::chrono::microseconds>(delay).count();
            return next;
        }
        const auto status = wait_(upload_.get(), ticket_, 0);
        need(status == 0 || status == 1, "could not poll native upload ticket");
        if (!status) return {};
        need(bind_(upload_.get(), ticket_), "cannot bind completed upload for device consumption");
        Step done;
        done.completed = ticket_;
        ticket_ = 0;
        phase_ = Phase::idle;
        direct_ = nullptr;
        return done;
    }

    // Caller has waited for the source's compute or DMA signal. An object may
    // use fragmented reserved pages; the copy must cover every byte exactly.
    size_t device_copy(size_t id, const Target &target, const R10StagePool &pool, bool materialize) {
        std::vector<std::pair<Address, Address>> work;
        size_t bytes = 0;
        for (const auto &part : target.at(id)) {
            const auto &a = part.where;
            need(a.object_offset == bytes, "native scatter parts do not form the exact object");
            pool.pieces(id, a.object_offset, a.bytes, [&](size_t page, size_t offset, size_t n) {
                const Address arena{stage_.get(), arena_base_ + page, n, a.object_offset + offset};
                const Address native{a.buffer, a.base + offset, n, a.object_offset + offset};
                work.emplace_back(materialize ? native : arena, materialize ? arena : native);
            });
            bytes += a.bytes;
        }
        need(!work.empty() && bytes == pool.get(id).bytes, "native scatter did not cover the object");
        TensorMeta meta{ggml_init({2 * work.size() * ggml_tensor_overhead(), nullptr, true}), ggml_free};
        need(bool(meta), "could not allocate D2D descriptors");
        for (const auto &[src, dst] : work) {
            auto *a = slice(meta.get(), src.buffer, src.base, src.bytes);
            auto *b = slice(meta.get(), dst.buffer, dst.base, dst.bytes);
            ggml_backend_tensor_copy_async(scatter_.get(), scatter_.get(), a, b);
        }
        ggml_backend_synchronize(scatter_.get());
        return bytes;
    }

private:
    enum class Phase { idle, disk, diagnostic_hold, device };
    static size_t maximum(const Plan &p) {
        size_t n = 0;
        for (const auto &o : p.objects) n = std::max(n, o.bytes);
        need(n && n % DiskRead::alignment == 0, "no aligned objects in native transfer catalogue");
        return n;
    }
    template<typename F> static F function(ggml_backend_dev_t device, const char *name) {
        const auto raw = ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), name);
        F typed{};
        static_assert(sizeof(typed) == sizeof(raw));
        std::memcpy(&typed, &raw, sizeof(typed));
        need(typed != nullptr, name);
        return typed;
    }
    static ggml_tensor *slice(ggml_context *meta, ggml_backend_buffer_t buffer, uintptr_t at, size_t n) {
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
        const auto capacity = ggml_backend_buffer_get_size(buffer);
        need(!ggml_backend_buffer_is_host(buffer) && at >= base && at - base <= capacity && n && n <= capacity - (at - base) &&
             (at | n) % 4 == 0, "invalid native device slice");
        auto *view = ggml_new_tensor_1d(meta, GGML_TYPE_I8, int64_t(n));
        need(view && ggml_backend_tensor_alloc(buffer, view, reinterpret_cast<void *>(at)) == GGML_STATUS_SUCCESS,
             "could not allocate native device view");
        return view;
    }

    Step issue(const R10StagePool &pool) {
        size_t bytes = 0;
        if (direct_) {
            for (const auto &part : direct_->at(id_)) {
                const auto &a = part.where;
                need(a.object_offset == bytes && copy_(upload_.get(), part.tensor, host_ + bytes, part.tensor_offset, a.bytes),
                     "private lane rejected a native target upload");
                bytes += a.bytes;
            }
        } else {
            pool.pieces(id_, 0, pool.get(id_).bytes, [&](size_t arena, size_t off, size_t n) {
                need(off == bytes && copy_(upload_.get(), stage_tensor_, host_ + off, arena, n), "staging lane rejected an upload");
                bytes += n;
            });
        }
        need(bytes == pool.get(id_).bytes && (ticket_ = submit_(upload_.get())), "missing payload or device completion ticket");
        phase_ = Phase::device;
        Step started;
        started.submitted = ticket_;
        return started;
    }

    const Plan &plan_;
    fs::path root_;
    ggml_backend_buffer_type_t device_type_;
    Backend upload_, scatter_;
    Buffer stage_, pinned_;
    TensorMeta tensor_meta_;
    DiskRead disk_;
    bool (*copy_)(ggml_backend_t, ggml_tensor *, const void *, size_t, size_t);
    uint64_t (*submit_)(ggml_backend_t);
    int (*wait_)(ggml_backend_t, uint64_t, uint64_t);
    bool (*bind_)(ggml_backend_t, uint64_t);
    std::map<size_t, Bytes> ram_;
    size_t ram_bytes_ = 0, id_ = 0;
    Phase phase_ = Phase::idle;
    const Target *direct_ = nullptr;
    ggml_tensor *stage_tensor_ = nullptr;
    uint8_t *host_ = nullptr;
    uintptr_t arena_base_ = 0;
    uint64_t ticket_ = 0;
    std::chrono::milliseconds soft_hold_{};
    DiskRead::Clock::time_point hold_start_{};
};
}
