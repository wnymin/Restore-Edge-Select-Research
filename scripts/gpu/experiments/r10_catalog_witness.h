#pragma once

#include "r10_catalog_source.h"
#include "r10_vk_trace_abi.h"

#include <cstring>

namespace r10cat {
// Explicit layer observes the existing timeline signal. Querying it does not
// submit a device command or substitute a host timer for device readiness.
class TimelineWitness {
public:
    TimelineWitness(const fs::path &library) : module_(LoadLibraryW(fs::absolute(library).c_str())) {
        need(module_, "cannot open explicit Vulkan witness");
        try {
            load(arm_, "r10_trace_arm");
            load(wait_, "r10_trace_wait");
        } catch (...) {
            FreeLibrary(module_);
            throw;
        }
    }
    ~TimelineWitness() { FreeLibrary(module_); }
    TimelineWitness(const TimelineWitness &) = delete;
    TimelineWitness &operator=(const TimelineWitness &) = delete;

    template<typename Call> auto observe(uint64_t id, Call &&call) {
        need(id && arm_(id) > 0, "cannot arm timeline witness");
        struct Reset { R10TraceArm arm; ~Reset() { arm(0); } } scope{arm_};
        return call();
    }
    std::pair<bool, R10TraceRecord> poll(uint64_t id) const {
        R10TraceRecord record{};
        const auto rc = wait_(id, 0, &record);
        need(record.queue && record.timeline_value && (rc == R10_TRACE_OK || rc == R10_TRACE_WAIT_TIMEOUT),
             "timeline witness has no valid device signal");
        return {rc == R10_TRACE_OK, record};
    }

private:
    template<typename F> void load(F &fn, const char *name) {
        const auto addr = GetProcAddress(module_, name);
        static_assert(sizeof(fn) == sizeof(addr));
        std::memcpy(&fn, &addr, sizeof(fn));
        need(fn, name);
    }
    HMODULE module_;
    R10TraceArm arm_{};
    R10TraceWait wait_{};
};
}
