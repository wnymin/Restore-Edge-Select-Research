#pragma once

#include "r10_catalog.h"

#include <chrono>
#include <optional>
#include <system_error>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace r10cat {
// One pending disk object with its own aligned storage. GPU commands remain on
// the executor's thread; the transfer lane stays occupied during disk and upload.
class DiskRead {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr size_t alignment = 4096;
    struct Completion { size_t bytes; int64_t observed_us; };

    explicit DiskRead(size_t capacity) : capacity_(capacity) {
        need(capacity && capacity % alignment == 0 && capacity <= MAXDWORD, "invalid aligned read capacity");
        event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!event_) fail(GetLastError(), "creating SSD completion event");
        bounce_ = static_cast<uint8_t *>(VirtualAlloc(nullptr, capacity, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        if (!bounce_) {
            const auto code = GetLastError();
            CloseHandle(event_);
            fail(code, "allocating aligned SSD bounce buffer");
        }
        // VirtualAlloc guarantees Windows page alignment for this allocation.
    }
    ~DiskRead() {
        if (file_ != INVALID_HANDLE_VALUE) {
            CancelIoEx(file_, &overlapped_);
            DWORD ignored = 0;
            GetOverlappedResult(file_, &overlapped_, &ignored, TRUE);
            CloseHandle(file_);
        }
        VirtualFree(bounce_, 0, MEM_RELEASE);
        CloseHandle(event_);
    }
    DiskRead(const DiskRead &) = delete;
    DiskRead &operator=(const DiskRead &) = delete;

    void start(const fs::path &path, size_t n) {
        need(file_ == INVALID_HANDLE_VALUE && n && n <= capacity_ && n % alignment == 0 &&
             reinterpret_cast<uintptr_t>(bounce_) % alignment == 0, "busy or unaligned SSD read");
        ready_ = 0;
        start_ = Clock::now();
        file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) fail(GetLastError(), "opening unbuffered SSD object");
        try {
            LARGE_INTEGER len{};
            if (!GetFileSizeEx(file_, &len)) fail(GetLastError(), "checking SSD object length");
            need(len.QuadPart == LONGLONG(n), "unbuffered SSD object is incomplete");
            if (!ResetEvent(event_)) fail(GetLastError(), "resetting SSD read event");
            overlapped_ = {};
            overlapped_.hEvent = event_;
            if (!ReadFile(file_, bounce_, DWORD(n), nullptr, &overlapped_)) {
                const auto code = GetLastError();
                if (code != ERROR_IO_PENDING) fail(code, "submitting unbuffered read");
            }
            expected_ = n;
        } catch (...) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            throw;
        }
    }

    std::optional<Completion> poll(bool wait = false) {
        need(file_ != INVALID_HANDLE_VALUE && expected_, "no pending SSD object");
        DWORD got = 0;
        if (!GetOverlappedResult(file_, &overlapped_, &got, wait ? TRUE : FALSE)) {
            const auto code = GetLastError();
            if (!wait && code == ERROR_IO_INCOMPLETE) return std::nullopt;
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            expected_ = 0;
            fail(code, "completing unbuffered SSD read");
        }
        CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
        const auto expected = expected_;
        expected_ = 0;
        need(got == expected, "actual SSD read is shorter than sealed object");
        ready_ = got;
        return Completion{got, std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start_).count()};
    }

    const uint8_t *payload(size_t n) const {
        need(n && ready_ == n && file_ == INVALID_HANDLE_VALUE, "SSD bytes requested before verified completion");
        return bounce_;
    }

private:
    [[noreturn]] static void fail(DWORD code, const char *what) { throw std::system_error(int(code), std::system_category(), what); }
    size_t capacity_, expected_ = 0, ready_ = 0;
    uint8_t *bounce_ = nullptr;
    HANDLE event_ = nullptr, file_ = INVALID_HANDLE_VALUE;
    OVERLAPPED overlapped_{};
    Clock::time_point start_{};
};
}
