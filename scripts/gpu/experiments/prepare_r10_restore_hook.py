"""Add the experimental restore-span API to the extracted pinned llama.cpp tree.

Without --apply this checks every edit without modifying any file. The API is
for an empty sequence owned by the calling scheduler. It neither synchronizes
nor dispatches a write; a validated list of native tensor writes is returned.
"""

import argparse
from pathlib import Path

from scripts.gpu.experiments.build_r10_native_vulkan import SOURCE

API = """    // R10 experimental use only. Successful prepare gives the destination sequence
    // metadata before its tensor payload is visible. The caller owns the context,
    // keeps src alive, installs all spans, and orders consumption after the upload.
    struct llama_r10_restore_span {
        struct ggml_tensor * tensor;
        size_t offset;
        const uint8_t * src;
        size_t size;
    };

    LLAMA_API size_t llama_r10_prepare_seq_restore(
            struct llama_context * ctx, const uint8_t * src, size_t size,
            llama_seq_id empty_seq, struct llama_r10_restore_span * spans,
            size_t span_capacity, size_t * span_count);

"""

DECL = """    size_t r10_prepare_seq_restore(llama_seq_id seq_id, const uint8_t * src, size_t size,
            llama_r10_restore_span * spans, size_t span_capacity, size_t * span_count);
"""

METHOD = """size_t llama_context::r10_prepare_seq_restore(llama_seq_id seq_id, const uint8_t * src, size_t size,
        llama_r10_restore_span * spans, size_t span_capacity, size_t * span_count) {
    if (!span_count) {
        return 0;
    }
    *span_count = 0;
    if (!memory || !src || !spans || seq_id < 0 || uint32_t(seq_id) >= n_seq_max() || memory->seq_pos_max(seq_id) >= 0) {
        return 0;
    }
    try {
        llama_io_read_host io(src, size, false);
        uint32_t magic_read;
        llama_seq_id source_seq;
        io.read(&magic_read, sizeof(magic_read));
        io.read(&source_seq, sizeof(source_seq));
        if (magic_read != io_magic) {
            return 0;
        }
        const size_t used = state_seq_read_data(io, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (used == size && io.copy_r10_spans(spans, span_capacity, span_count) && *span_count) {
            return used;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error planning restore: %s\\n", __func__, err.what());
    }
    // Planning mutates the destination's metadata even if the reader fails.
    memory->seq_rm(seq_id, -1, -1);
    *span_count = 0;
    return 0;
}

"""

C_API = """size_t llama_r10_prepare_seq_restore(llama_context * ctx, const uint8_t * src, size_t size,
        llama_seq_id empty_seq, llama_r10_restore_span * spans, size_t span_capacity, size_t * span_count) {
    if (!ctx) {
        if (span_count) { *span_count = 0; }
        return 0;
    }
    return ctx->r10_prepare_seq_restore(empty_seq, src, size, spans, span_capacity, span_count);
}

"""

READER_ORIGINAL = """    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
"""
READER_UPDATED = """    llama_io_read_host(const uint8_t * p, size_t len, bool flush = true) : ptr(p), buf_size(len), flush(flush) {}

    ~llama_io_read_host() {
        if (!flush) {
            return;
        }
        // flush the reads
"""
READER_METHOD_ANCHOR = """    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file"""
READER_METHOD = """    const bool flush;

public:
    bool copy_r10_spans(llama_r10_restore_span * out, size_t capacity, size_t * count) const {
        if (rinfos.size() > capacity) {
            return false;
        }
        for (size_t i = 0; i < rinfos.size(); ++i) {
            const auto & r = rinfos[i];
            out[i] = {r.tensor, r.offset, r.ptr, r.size};
        }
        *count = rinfos.size();
        return true;
    }

private:
"""


def insert(before, anchor, addition):
    if before.count(anchor) != 1:
        raise ValueError(f"Unexpected pinned source around {anchor[:100]!r}")
    return before if addition + anchor in before else before.replace(anchor, addition + anchor)


def patch(source):
    api = source / "include/llama.h"
    context_h = source / "src/llama-context.h"
    context_cpp = source / "src/llama-context.cpp"
    plans = {}
    before = api.read_text(encoding="utf-8")
    plans[api] = insert(before, "    LLAMA_API size_t llama_state_seq_set_data_ext(\n", API)
    before = context_h.read_text(encoding="utf-8")
    plans[context_h] = insert(before, "    size_t state_seq_set_data(llama_seq_id seq_id,", DECL)
    before = context_cpp.read_text(encoding="utf-8")
    planned = insert(before, "bool llama_context::state_load_file(", METHOD)
    planned = insert(planned, "size_t llama_state_seq_load_file(", C_API)
    planned = insert(planned, READER_METHOD_ANCHOR, READER_METHOD)
    if READER_UPDATED not in planned:
        if planned.count(READER_ORIGINAL) != 1:
            raise ValueError("Unexpected pinned host-reader constructor")
        planned = planned.replace(READER_ORIGINAL, READER_UPDATED)
    plans[context_cpp] = planned
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plans = patch(SOURCE)
    for path, text in plans.items():
        differs = path.read_text(encoding="utf-8") != text
        if args.apply and differs:
            path.write_text(text, encoding="utf-8", newline="\n")
        print(f"{'updated' if args.apply and differs else 'verified'}: {path}")


if __name__ == "__main__":
    main()
