"""Opt-in Vulkan queue hooks for uploads to real llama.cpp tensors.

The dedicated upload backend records only DMA commands from mapped host storage.
A transfer-queue timeline semaphore is observed directly. An explicit compute
queue wait supplies device visibility before the caller submits any consumer.
"""
import argparse

from scripts.gpu.experiments.build_r10_native_vulkan import SOURCE
from scripts.gpu.experiments.prepare_r10_restore_hook import insert

LEGACY_SHARING = """    // R10 only: shared queue-family ownership for the opt-in upload path.
    std::array<uint32_t, 2> r10_queue_families{};
    if (getenv("GGML_VK_R10_CONCURRENT_BUFFERS") && device->compute_queue && device->transfer_queue &&
        device->compute_queue->queue_family_index != device->transfer_queue->queue_family_index) {
        r10_queue_families = {device->compute_queue->queue_family_index, device->transfer_queue->queue_family_index};
        buffer_create_info.setSharingMode(vk::SharingMode::eConcurrent);
        buffer_create_info.setQueueFamilyIndices(r10_queue_families);
    }

"""

SHARING = """// Keep the queue-family array alive through createBuffer. Keeping this small
// function separate also leaves the pinned MinGW allocator's control flow intact.
static vk::Buffer ggml_vk_r10_create_buffer(vk_device & device, vk::BufferCreateInfo & info) {
    if (getenv("GGML_VK_R10_CONCURRENT_BUFFERS") && device->compute_queue && device->transfer_queue &&
        device->compute_queue->queue_family_index != device->transfer_queue->queue_family_index) {
        const std::array<uint32_t, 2> families{device->compute_queue->queue_family_index,
                                             device->transfer_queue->queue_family_index};
        info.setSharingMode(vk::SharingMode::eConcurrent);
        info.setQueueFamilyIndices(families);
        return device->device.createBuffer(info);
    }
    return device->device.createBuffer(info);
}

"""

PROCS = """// The following hooks are private to R10 and require a dedicated upload
// backend; the caller serializes access to that backend and allocates the
// model's destination in a context with no overlapping tensor accesses.
static bool ggml_backend_vk_r10_copy(ggml_backend_t backend, ggml_tensor * tensor, const void * src,
                                    size_t off, size_t bytes) {
    if (!ggml_backend_is_vk(backend) || !tensor || !tensor->buffer || !src || !bytes ||
        off > ggml_nbytes(tensor) || bytes > ggml_nbytes(tensor) - off) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
    if (!ctx->device->async_use_transfer_queue || !ctx->compute_ctx.expired() || ctx->submit_pending ||
        !getenv("GGML_VK_R10_CONCURRENT_BUFFERS") ||
        tensor->buffer->buft != ggml_backend_vk_get_default_buffer_type(backend)) {
        return false;
    }
    auto * bctx = static_cast<ggml_backend_vk_buffer_context *>(tensor->buffer->context);
    auto & dst = bctx->dev_buffer;
    const size_t dst_offset = vk_tensor_offset(tensor) + tensor->view_offs + off;
    vk_buffer pinned;
    size_t pinned_offset = 0;
    ggml_vk_host_get(ctx->device, src, pinned, pinned_offset);
    if (!pinned || (pinned_offset | dst_offset | bytes) % 4 ||
        pinned_offset > pinned->size || bytes > pinned->size - pinned_offset) {
        return false;
    }
    auto cpy = ggml_vk_get_transfer_ctx(ctx);
    return ggml_vk_buffer_write_2d_async(cpy, dst, dst_offset, src, bytes, bytes, bytes, 1);
}

static uint64_t ggml_backend_vk_r10_submit(ggml_backend_t backend) {
    if (!ggml_backend_is_vk(backend)) {
        return 0;
    }
    auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
    if (!ctx->compute_ctx.expired() || ctx->submit_pending || !ggml_vk_submit_transfer_ctx(ctx)) {
        return 0;
    }
    return ctx->transfer_semaphore.value;
}

static int ggml_backend_vk_r10_wait(ggml_backend_t backend, uint64_t value, uint64_t timeout_ns) {
    if (!ggml_backend_is_vk(backend) || !value) {
        return -1;
    }
    auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
    if (!ctx->transfer_ctx.expired() || !ctx->compute_ctx.expired() || ctx->submit_pending ||
        value != ctx->transfer_semaphore.value) {
        return -1;
    }
    const vk::Semaphore semaphore = ctx->transfer_semaphore.s;
    const vk::SemaphoreWaitInfo wait{{}, 1, &semaphore, &value};
    const auto result = ctx->device->device.waitSemaphores(wait, timeout_ns);
    if (result == vk::Result::eSuccess) {
        ggml_vk_graph_cleanup(ctx);
        return 1;
    }
    return result == vk::Result::eTimeout ? 0 : -1;
}

static bool ggml_backend_vk_r10_bind(ggml_backend_t backend, uint64_t value) {
    if (!ggml_backend_is_vk(backend) || !value) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
    if (value != ctx->transfer_semaphore.value || !ctx->transfer_ctx.expired()) {
        return false;
    }
    const vk::Semaphore s = ctx->transfer_semaphore.s;
    const vk::TimelineSemaphoreSubmitInfo timeline{1, &value, 0, nullptr};
    const vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands;
    vk::SubmitInfo submit{1, &s, &stage, 0, nullptr, 0, nullptr};
    submit.setPNext(&timeline);
    ctx->device->compute_queue->handle->submit({ submit }, {});
    return true;
}

static void * ggml_backend_vk_r10_get_proc(ggml_backend_reg_t, const char * name) {
    if (strcmp(name, "ggml_backend_vk_r10_copy") == 0) return reinterpret_cast<void *>(ggml_backend_vk_r10_copy);
    if (strcmp(name, "ggml_backend_vk_r10_submit") == 0) return reinterpret_cast<void *>(ggml_backend_vk_r10_submit);
    if (strcmp(name, "ggml_backend_vk_r10_wait") == 0) return reinterpret_cast<void *>(ggml_backend_vk_r10_wait);
    if (strcmp(name, "ggml_backend_vk_r10_bind") == 0) return reinterpret_cast<void *>(ggml_backend_vk_r10_bind);
    return nullptr;
}

"""


def plan(path):
    before = path.read_text(encoding="utf-8")
    after = before.replace(LEGACY_SHARING, "")
    after = insert(after, "static vk_buffer ggml_vk_create_buffer(vk_device& device,", SHARING)
    old = "    buf->buffer = device->device.createBuffer(buffer_create_info);"
    new = "    buf->buffer = ggml_vk_r10_create_buffer(device, buffer_create_info);"
    if new not in after:
        if after.count(old) != 1:
            raise ValueError("Unexpected Vulkan buffer allocation call")
        after = after.replace(old, new)
    registry = "static const struct ggml_backend_reg_i ggml_backend_vk_reg_i = {"
    if "static bool ggml_backend_vk_r10_copy(" not in after:
        after = insert(after, registry, PROCS)
    remove = "    /* .get_proc_address = */ NULL,\n"
    add = "    /* .get_proc_address = */ ggml_backend_vk_r10_get_proc,\n"
    if add not in after:
        if after.count(remove) != 1:
            raise ValueError("Unexpected Vulkan registry proc slot")
        after = after.replace(remove, add)
    return before, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    path = SOURCE / "ggml/src/ggml-vulkan/ggml-vulkan.cpp"
    before, after = plan(path)
    if args.apply and before != after:
        path.write_text(after, encoding="utf-8", newline="\n")
    print(f"{'updated' if args.apply and before != after else 'verified'}: {path}")


if __name__ == "__main__":
    main()
