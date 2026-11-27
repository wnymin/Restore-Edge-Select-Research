#pragma once
#include <cstdint>

struct R10TraceRecord {
    int64_t submitted_qpc;
    int64_t observed_ready_qpc;
    uintptr_t queue;
    uint64_t timeline_value;
};

enum { R10_TRACE_OK = 0, R10_TRACE_NO_SIGNAL = 1, R10_TRACE_WAIT_TIMEOUT = 2, R10_TRACE_WAIT_ERROR = 3 };
using R10TraceArm = int (*)(uint64_t arm);
using R10TraceWait = int (*)(uint64_t arm, uint32_t timeout_ms, R10TraceRecord *out);
