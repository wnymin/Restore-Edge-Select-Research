#pragma once

#include "r10_catalog.h"

#include <limits>

namespace r10cat {
// Capture address and buffer now: the next prepare may change ggml tensor
// descriptors. Copies from an older active object must use this saved address.
struct Address {
    ggml_backend_buffer_t buffer;
    uintptr_t base;
    size_t bytes, object_offset;
    bool operator==(const Address &other) const {
        return buffer == other.buffer && base == other.base && bytes == other.bytes && object_offset == other.object_offset;
    }
};

struct TargetPart {
    Address where;
    ggml_tensor *tensor;
    size_t tensor_offset;
};
using Target = std::map<size_t, std::vector<TargetPart>>;

inline size_t prepare_spans(llama_context *ctx, const Bytes &image, size_t depth, Shape &spans) {
    size_t n = 0;
    need(depth && llama_r10_prepare_seq_restore(ctx, image.data(), image.size(), 0, spans.data(), spans.size(), &n) == image.size() &&
         n && n <= spans.size() && llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == llama_pos(depth - 1),
         "invalid native restore plan");
    return n;
}

inline Target bind_target(llama_context *ctx, const Meta &meta, ggml_backend_buffer_type_t device_type) {
    Shape spans{};
    const auto n = prepare_spans(ctx, meta.skeleton, meta.depth, spans);
    need(n == meta.regions.size(), "restore plan no longer has the sealed number of spans");
    Target target;
    for (size_t i = 0; i < n; ++i) {
        const auto &s = spans[i];
        const auto &r = meta.regions[i];
        need(s.tensor && s.tensor->buffer && s.tensor->data && r.tensor == ggml_get_name(s.tensor) && s.size == r.bytes &&
             r.source <= meta.skeleton.size() && s.src == meta.skeleton.data() + r.source &&
             ggml_backend_buffer_get_type(s.tensor->buffer) == device_type,
             "restore plan no longer maps to the sealed native tensor");
        uintptr_t base = reinterpret_cast<uintptr_t>(s.tensor->data);
        need(s.offset <= std::numeric_limits<uintptr_t>::max() - base &&
             s.size <= std::numeric_limits<uintptr_t>::max() - base - s.offset, "native tensor address overflow");
        size_t offset = 0;
        for (const auto &part : r.parts) {
            need(part.bytes <= s.size - offset, "invalid native target object part");
            const auto item_offset = s.offset + offset;
            target[part.object].push_back({{s.tensor->buffer, base + item_offset, part.bytes, part.offset}, s.tensor, item_offset});
            offset += part.bytes;
        }
        need(offset == s.size, "native target tensor was not covered");
    }
    return target;
}

inline bool same_native_bytes(const std::vector<TargetPart> &left, const std::vector<TargetPart> &right) {
    return !left.empty() && left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin(), [](const auto &a, const auto &b) { return a.where == b.where; });
}
}
