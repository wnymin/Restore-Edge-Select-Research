"""Work around a reproduced MinGW/MSVCRT formatter crash in two core units."""

import argparse

from scripts.gpu.experiments.build_r10_native_vulkan import SOURCE
from scripts.gpu.experiments.prepare_r10_restore_hook import insert

HEADER = """#pragma once
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

#if defined(_WIN32) && defined(__MINGW32__)
// Avoid MinGW's __mingw_vsnprintf adapter on the pinned Windows build.
// MSVCRT _vsnprintf can truncate without a terminator and returns -1;
// _vscprintf supplies the required length for the caller's sizing pass.
static inline int r10_vsnprintf(char *dst, size_t cap, const char *fmt, va_list args) {
    if (!cap) return _vscprintf(fmt, args);
    va_list copy;
    va_copy(copy, args);
    const int wrote = _vsnprintf(dst, cap, fmt, args);
    if (wrote >= 0 && (size_t)wrote < cap) {
        va_end(copy);
        return wrote;
    }
    dst[cap - 1] = 0;
    const int required = _vscprintf(fmt, copy);
    va_end(copy);
    if (required < 0) dst[0] = 0;
    return required;
}
#else
#define r10_vsnprintf vsnprintf
#endif
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    plans = {}
    for where, anchor, include in (
        ("src/llama-impl.cpp", "struct llama_logger_state {", '../ggml/src/r10-win-format.h'),
        ("ggml/src/ggml.c", "struct ggml_logger_state {", 'r10-win-format.h'),
    ):
        file = SOURCE / where
        text = file.read_text(encoding="utf-8")
        # The first C use precedes the logger. Insert before C's first exported function instead.
        if file.name == "ggml.c":
            anchor = "#include \"ggml.h\"\n"
        before = text
        if "r10_vsnprintf(" not in text:
            if text.count("vsnprintf(") != 4:
                raise ValueError(f"Unexpected format call sites in {where}")
            text = text.replace("vsnprintf(", "r10_vsnprintf(")
        plans[file] = (before, insert(text, anchor, f'#include "{include}"\n\n'))
    header = SOURCE / "ggml/src/r10-win-format.h"
    plans[header] = (header.read_text(encoding="utf-8") if header.exists() else "", HEADER)
    for path, (before, after) in plans.items():
        if args.apply and before != after:
            path.write_text(after, encoding="utf-8", newline="\n")
        print(f"{'updated' if args.apply and before != after else 'verified'}: {path}")


if __name__ == "__main__":
    main()
