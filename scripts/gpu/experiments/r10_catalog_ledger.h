#pragma once

#include "r10_stage_pool.h"

namespace r10cat {
// Physical readiness for the single-lane executor. A published lease can own
// charged pages without those pages containing the object: the sole copy may
// still reside in the active execution allocation. The caller performs the
// byte copies and supplies actual device completion before changing readiness.
class PayloadLedger {
public:
    enum class Source { active, staged };
    struct Transfer { size_t request, object; bool direct; uint64_t dma = 0; };

    PayloadLedger(size_t capacity, const std::vector<size_t> &sizes) : pool_(capacity, sizes) {}
    const R10StagePool &pool() const { return pool_; }
    const std::set<size_t> &aliases() const { return aliases_; }

    void reserve(size_t request, const std::vector<size_t> &deps, const std::set<size_t> &active_kv) {
        check(!deps.empty() && pool_.reserve(request, deps), "device pages differ from controller admission");
        for (const auto id : active_kv) {
            check(phase_ == Phase::computing && local_.count(id) &&
                  std::find(deps.begin(), deps.end() - 1, id) != deps.end() - 1, "active loan is not matching immutable KV");
        }
        for (const auto id : deps) {
            const bool existing = (phase_ == Phase::assembly && local_.count(id)) || active_kv.count(id);
            if (existing && pool_.get(id).state == R10StagePool::State::reserved) mark_alias(id);
        }
    }

    // These older active aliases must be copied into their charged pages before
    // turning the native execution allocation over to a different request. The
    // caller obtains keep_kv only from exact matching native address maps.
    std::vector<size_t> before_switch(const std::set<size_t> &keep_kv) const {
        check(phase_ == Phase::idle && (!moving_ || !move_.direct), "old execution allocation is still in use");
        std::vector<size_t> required;
        for (auto id : aliases_) if (!keep_kv.count(id)) required.push_back(id);
        return required;
    }

    // Call only when the complete native copy is visible. This never changes
    // admission, total charged pages or what the scheduler already saw as ready.
    void materialized(size_t id) {
        check(pool_.get(id).state == R10StagePool::State::ready && aliases_.erase(id) == 1,
              "materialization requires a ready alias and its charged pages");
    }

    void bind_head(size_t request, std::vector<size_t> deps, const std::set<size_t> &keep_kv) {
        check(before_switch(keep_kv).empty(), "switch would overwrite a lease backed only by the old active allocation");
        for (auto id : keep_kv) check(aliases_.count(id) && !deps.empty() &&
              std::find(deps.begin(), deps.end() - 1, id) != deps.end() - 1, "cannot carry an unrelated or recurrent alias into a new head");
        local_ = keep_kv;
        deps_ = std::move(deps);
        request_ = request;
        phase_ = Phase::assembly;
    }

    Transfer dispatch(size_t request, size_t object, bool demand) {
        check(!moving_, "serial device lane already in use");
        const auto state = pool_.get(object).state;
        if (state == R10StagePool::State::absent) {
            check(demand && phase_ == Phase::assembly && request_ == request && !local_.count(object) &&
                  std::find(deps_.begin(), deps_.end(), object) != deps_.end(), "private demand must own the idle native target");
        } else {
            check(state == R10StagePool::State::reserved, "prefetch/demand does not own an uploadable object");
        }
        moving_ = true;
        move_ = {request, object, state == R10StagePool::State::absent, 0};
        return move_;
    }

    void submitted(uint64_t dma) {
        check(moving_ && !move_.dma && dma && dma > last_fence_, "invalid upload ticket");
        if (!move_.direct) pool_.issued(move_.object, dma);
        move_.dma = last_fence_ = dma;
    }

    Transfer landed(uint64_t dma) {
        check(moving_ && dma && dma == move_.dma, "completion does not belong to the native transfer lane");
        if (move_.direct) {
            check(phase_ == Phase::assembly && request_ == move_.request && local_.insert(move_.object).second,
                  "duplicate private completion or changed native head");
            if (pool_.get(move_.object).state == R10StagePool::State::reserved) mark_alias(move_.object);
        } else pool_.landed(move_.object, dma);
        moving_ = false;
        return move_;
    }

    Source source(size_t id) const {
        check(phase_ == Phase::assembly && std::find(deps_.begin(), deps_.end(), id) != deps_.end(), "unknown target object");
        if (local_.count(id)) return Source::active;
        check(pool_.get(id).state == R10StagePool::State::ready && !aliases_.count(id), "native target object lacks ready bytes");
        return Source::staged;
    }

    void scattered(size_t id) {
        check(source(id) == Source::staged && local_.insert(id).second, "cannot report a duplicate or absent scatter");
    }

    bool recurrent_must_preserve(bool release, bool has_suffix) const {
        check(phase_ == Phase::assembly, "only an assembled head has a recurrent dependency");
        if (deps_.empty()) return false;
        const auto id = deps_.back();
        return has_suffix && aliases_.count(id) && pool_.get(id).refs > size_t(release);
    }

    void begin_compute(size_t request, bool release, bool has_suffix) {
        check(phase_ == Phase::assembly && request == request_ &&
              (!moving_ || !move_.direct) && local_.size() == deps_.size(), "incomplete native restore");
        check(release == pool_.matches(request, deps_) && !recurrent_must_preserve(release, has_suffix),
              "compute would release the wrong lease or mutate a shared checkpoint");
        if (release) {
            pool_.release(request);
            for (auto id : deps_) if (!pool_.get(id).refs) aliases_.erase(id);
        }
        if (has_suffix && !deps_.empty()) local_.erase(deps_.back());
        phase_ = Phase::computing;
    }

    void computed(size_t request) {
        check(phase_ == Phase::computing && request == request_, "wrong native compute completion");
        phase_ = Phase::idle;
    }

    bool drained() const { return phase_ == Phase::idle && !moving_ && !pool_.charged(); }

private:
    enum class Phase { idle, assembly, computing };
    static void check(bool ok, const char *what) { if (!ok) throw std::logic_error(what); }
    void mark_alias(size_t id) {
        check(!aliases_.count(id), "duplicate native alias");
        pool_.borrow_active(id);
        aliases_.insert(id);
    }
    R10StagePool pool_;
    std::set<size_t> aliases_, local_;
    std::vector<size_t> deps_;
    size_t request_ = 0;
    uint64_t last_fence_ = 0;
    Transfer move_{};
    bool moving_ = false;
    Phase phase_ = Phase::idle;
};
}
