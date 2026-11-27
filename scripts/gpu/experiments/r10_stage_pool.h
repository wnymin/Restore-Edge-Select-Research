#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

// One page view over a fixed-size device allocation. The caller owns the
// backing storage and must bind/copy from it before leasing pages again.
class R10StagePool {
public:
    static constexpr size_t page = 64 * 1024;
    enum class State { absent, reserved, flying, ready };
    struct Object {
        size_t bytes = 0;
        std::vector<size_t> pages;
        size_t refs = 0;
        State state = State::absent;
        uint64_t fence = 0;
    };

    explicit R10StagePool(size_t cap, const std::vector<size_t> &sizes) : capacity_(cap) {
        require(cap && cap % page == 0, "stage capacity must be whole pages");
        for (size_t p = 0; p < cap / page; ++p) free_.insert(p);
        for (size_t bytes : sizes) {
            require(bytes && bytes % 4 == 0, "invalid device object size");
            Object obj;
            obj.bytes = bytes;
            objects_.push_back(std::move(obj));
        }
    }
    static size_t cost(size_t bytes) { return (bytes + page - 1) / page * page; }
    size_t charged() const { return capacity_ - free_.size() * page; }
    size_t capacity() const { return capacity_; }
    const Object &get(size_t id) const { return objects_.at(id); }

    bool reserve(size_t request, const std::vector<size_t> &deps) {
        require(!leases_.count(request), "already admitted request");
        require(std::set<size_t>(deps.begin(), deps.end()).size() == deps.size(), "duplicate object in a dependency list");
        size_t missing = 0;
        for (auto id : deps) if (!get(id).refs) missing += cost(get(id).bytes);
        if (missing > capacity_ - charged()) return false;
        for (auto id : deps) {
            auto &o = objects_.at(id);
            if (o.refs++ == 0) {
                o.state = State::reserved;
                for (size_t i = 0; i < cost(o.bytes) / page; ++i) {
                    auto p = free_.begin();
                    o.pages.push_back(*p);
                    free_.erase(p);
                }
            }
        }
        leases_.emplace(request, deps);
        return true;
    }

    void issued(size_t id, uint64_t fence) {
        auto &o = objects_.at(id);
        require(o.state == State::reserved && !flying_ && fence > last_fence_, "invalid or concurrent upload");
        o.state = State::flying;
        o.fence = fence;
        last_fence_ = fence;
        flying_ = true;
    }
    void landed(size_t id, uint64_t fence) {
        auto &o = objects_.at(id);
        require(flying_ && o.state == State::flying && o.fence == fence, "invalid upload completion identity");
        o.state = State::ready;
        flying_ = false;
    }
    // The owning executor proves that this identity is already accessible in
    // an active device allocation. Pages stay charged during this alias lease.
    void borrow_active(size_t id) {
        auto &o = objects_.at(id);
        require(o.state == State::reserved && o.refs, "cannot borrow a moving, absent or populated object");
        o.state = State::ready;
    }
    bool consumable(size_t request) const {
        auto lease = leases_.find(request);
        return lease != leases_.end() && std::all_of(lease->second.begin(), lease->second.end(),
            [&](size_t id) { return get(id).state == State::ready; });
    }
    bool matches(size_t request, const std::vector<size_t> &deps) const {
        const auto lease = leases_.find(request);
        return lease != leases_.end() && lease->second == deps;
    }
    void release(size_t request) {
        const auto &deps = leases_.at(request);
        for (auto id : deps) {
            const auto &o = get(id);
            require(o.state != State::flying || o.refs > 1, "cannot recycle the final lease of an unfinished transfer");
        }
        for (auto id : deps) {
            auto &o = objects_.at(id);
            if (--o.refs == 0) {
                free_.insert(o.pages.begin(), o.pages.end());
                o.pages.clear();
                o.state = State::absent;
                o.fence = 0;
            }
        }
        leases_.erase(request);
    }

    template<typename Callback> void pieces(size_t id, size_t offset, size_t n, Callback use) const {
        const auto &o = get(id);
        require(o.refs && offset <= o.bytes && n <= o.bytes - offset, "stage span out of bounds");
        size_t consumed = 0;
        while (n) {
            const size_t i = offset / page;
            const size_t start = o.pages.at(i) * page + offset % page;
            size_t chunk = std::min(n, page - offset % page);
            for (size_t j = i + 1; chunk < n && j < o.pages.size() && o.pages[j] == o.pages[j - 1] + 1; ++j) {
                chunk = std::min(n, chunk + page);
            }
            use(start, consumed, chunk);
            offset += chunk;
            consumed += chunk;
            n -= chunk;
        }
    }

private:
    static void require(bool ok, const char *why) { if (!ok) throw std::logic_error(why); }
    size_t capacity_;
    std::set<size_t> free_;
    std::vector<Object> objects_;
    std::map<size_t, std::vector<size_t>> leases_;
    uint64_t last_fence_ = 0;
    bool flying_ = false;
};
