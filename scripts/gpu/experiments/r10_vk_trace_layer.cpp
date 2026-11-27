// Experiment-only Vulkan layer. It observes the upload's existing timeline semaphore;
// it does not add commands, fences, waits on work queues, or layer state to the system.
#define VK_NO_PROTOTYPES
#include <vulkan/vk_layer.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "r10_vk_trace_abi.h"

namespace {
struct Instance {
    VkInstance handle;
    PFN_vkGetInstanceProcAddr get;
    PFN_GetPhysicalDeviceProcAddr get_physical;
};
struct Device {
    VkDevice handle;
    PFN_vkGetDeviceProcAddr get;
    PFN_vkQueueSubmit submit;
    PFN_vkQueueSubmit2 submit2, submit2_khr;
    PFN_vkWaitSemaphores wait;
    PFN_vkCreateComputePipelines create_pipelines;
};
struct Observation {
    uint64_t arm = 0;
    Device device{};
    VkSemaphore signal{};
    uint64_t value = 0;
    VkQueue queue{};
    int64_t submitted_qpc = 0;
};
std::mutex m;
std::condition_variable captured;
std::unordered_map<void *, Instance> instances;
std::unordered_map<void *, Device> devices;
Observation observation;
thread_local uint64_t transfer_arm = 0;

void *dispatch_key(const void *handle) {
    return *static_cast<void * const *>(handle);
}

int64_t qpc() {
    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);
    return now.QuadPart;
}

bool profile_pipelines() {
    static const bool enabled = std::getenv("R10_VK_PROFILE_PIPELINES") != nullptr;
    return enabled;
}

template <typename T> const T *from_chain(const void *start, VkStructureType kind) {
    for (auto *p = static_cast<const VkBaseInStructure *>(start); p; p = p->pNext) {
        if (p->sType == kind) return reinterpret_cast<const T *>(p);
    }
    return nullptr;
}

template <typename T> T *loader_link(const void *start, VkStructureType kind) {
    for (auto *p = static_cast<const VkBaseInStructure *>(start); p; p = p->pNext) {
        if (p->sType != kind) continue;
        auto *info = reinterpret_cast<const T *>(p);
        if (info->function == VK_LAYER_LINK_INFO) return const_cast<T *>(info);
    }
    return nullptr;
}

Device device_for(const void *handle) {
    std::lock_guard<std::mutex> lock(m);
    const auto it = devices.find(dispatch_key(handle));
    return it != devices.end() ? it->second : Device{};
}

VKAPI_ATTR VkResult VKAPI_CALL create_pipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                const VkComputePipelineCreateInfo *info, const VkAllocationCallbacks *alloc,
                                                VkPipeline *out) {
    const auto d = device_for(device);
    if (!d.create_pipelines) return VK_ERROR_INITIALIZATION_FAILED;
    const auto start = qpc();
    const auto rc = d.create_pipelines(device, cache, count, info, alloc, out);
    const auto end = qpc();
    std::fprintf(stderr, "r10_vk_pipeline,%lld,%lld,%lu,%u,%d\n", static_cast<long long>(start),
                 static_cast<long long>(end), static_cast<unsigned long>(GetCurrentThreadId()), count, int(rc));
    return rc;
}

void note_signal(const Device &d, VkQueue queue, VkSemaphore signal, uint64_t value) {
    if (!transfer_arm || !signal || !value || !d.wait) return;
    std::lock_guard<std::mutex> lock(m);
    if (observation.arm != transfer_arm || observation.signal) return;
    observation = {transfer_arm, d, signal, value, queue, qpc()};
    captured.notify_all();
}

VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue queue, uint32_t count, const VkSubmitInfo *items, VkFence fence) {
    const auto d = device_for(queue);
    if (!d.submit) return VK_ERROR_INITIALIZATION_FAILED;
    const auto outcome = d.submit(queue, count, items, fence);
    if (outcome != VK_SUCCESS || !transfer_arm) return outcome;
    for (uint32_t i = 0; i < count; ++i) {
        const auto &s = items[i];
        if (!s.commandBufferCount) continue;
        const auto *t = from_chain<VkTimelineSemaphoreSubmitInfo>(s.pNext, VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
        if (!t || t->signalSemaphoreValueCount != s.signalSemaphoreCount) continue;
        for (uint32_t k = 0; k < s.signalSemaphoreCount; ++k) note_signal(d, queue, s.pSignalSemaphores[k], t->pSignalSemaphoreValues[k]);
    }
    return outcome;
}

VkResult submit2_common(VkQueue queue, uint32_t count, const VkSubmitInfo2 *items, VkFence fence, bool khr) {
    const auto d = device_for(queue);
    const auto next = khr ? d.submit2_khr : d.submit2;
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;
    const auto outcome = next(queue, count, items, fence);
    if (outcome != VK_SUCCESS || !transfer_arm) return outcome;
    for (uint32_t i = 0; i < count; ++i) {
        const auto &s = items[i];
        if (!s.commandBufferInfoCount) continue;
        for (uint32_t k = 0; k < s.signalSemaphoreInfoCount; ++k) {
            const auto &signal = s.pSignalSemaphoreInfos[k];
            note_signal(d, queue, signal.semaphore, signal.value);
        }
    }
    return outcome;
}

VKAPI_ATTR VkResult VKAPI_CALL submit2(VkQueue q, uint32_t n, const VkSubmitInfo2 *s, VkFence f) {
    return submit2_common(q, n, s, f, false);
}
VKAPI_ATTR VkResult VKAPI_CALL submit2_khr(VkQueue q, uint32_t n, const VkSubmitInfo2 *s, VkFence f) {
    return submit2_common(q, n, s, f, true);
}

VKAPI_ATTR VkResult VKAPI_CALL create_instance(const VkInstanceCreateInfo *info, const VkAllocationCallbacks *alloc,
                                               VkInstance *out) {
    auto *link = loader_link<VkLayerInstanceCreateInfo>(info->pNext, VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO);
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;
    Instance next{VK_NULL_HANDLE, link->u.pLayerInfo->pfnNextGetInstanceProcAddr, link->u.pLayerInfo->pfnNextGetPhysicalDeviceProcAddr};
    const auto create = reinterpret_cast<PFN_vkCreateInstance>(next.get(nullptr, "vkCreateInstance"));
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    if (!create) return VK_ERROR_INITIALIZATION_FAILED;
    const auto outcome = create(info, alloc, out);
    if (outcome == VK_SUCCESS) {
        next.handle = *out;
        std::lock_guard<std::mutex> lock(m);
        instances[dispatch_key(*out)] = next;
    }
    return outcome;
}

VKAPI_ATTR void VKAPI_CALL destroy_instance(VkInstance instance, const VkAllocationCallbacks *alloc) {
    PFN_vkGetInstanceProcAddr get{};
    {
        std::lock_guard<std::mutex> lock(m);
        if (auto it = instances.find(dispatch_key(instance)); it != instances.end()) {
            get = it->second.get;
            instances.erase(it);
        }
    }
    if (get) reinterpret_cast<PFN_vkDestroyInstance>(get(instance, "vkDestroyInstance"))(instance, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL create_device(VkPhysicalDevice physical, const VkDeviceCreateInfo *info,
                                             const VkAllocationCallbacks *alloc, VkDevice *out) {
    auto *link = loader_link<VkLayerDeviceCreateInfo>(info->pNext, VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO);
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;
    const auto next = *link->u.pLayerInfo;
    VkInstance instance = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> lock(m);
        if (auto it = instances.find(dispatch_key(physical)); it != instances.end()) instance = it->second.handle;
    }
    if (!instance) return VK_ERROR_INITIALIZATION_FAILED;
    const auto create = reinterpret_cast<PFN_vkCreateDevice>(next.pfnNextGetInstanceProcAddr(instance, "vkCreateDevice"));
    link->u.pLayerInfo = next.pNext;
    if (!create) return VK_ERROR_INITIALIZATION_FAILED;
    const auto outcome = create(physical, info, alloc, out);
    if (outcome != VK_SUCCESS) return outcome;
    const auto g = next.pfnNextGetDeviceProcAddr;
    const auto d = *out;
    auto wait = reinterpret_cast<PFN_vkWaitSemaphores>(g(d, "vkWaitSemaphores"));
    if (!wait) wait = reinterpret_cast<PFN_vkWaitSemaphores>(g(d, "vkWaitSemaphoresKHR"));
    const Device device{d, g, reinterpret_cast<PFN_vkQueueSubmit>(g(d, "vkQueueSubmit")),
                        reinterpret_cast<PFN_vkQueueSubmit2>(g(d, "vkQueueSubmit2")),
                        reinterpret_cast<PFN_vkQueueSubmit2>(g(d, "vkQueueSubmit2KHR")), wait,
                        reinterpret_cast<PFN_vkCreateComputePipelines>(g(d, "vkCreateComputePipelines"))};
    std::lock_guard<std::mutex> lock(m);
    devices[dispatch_key(d)] = device;
    return outcome;
}

VKAPI_ATTR void VKAPI_CALL destroy_device(VkDevice device, const VkAllocationCallbacks *alloc) {
    const auto d = device_for(device);
    {
        std::lock_guard<std::mutex> lock(m);
        devices.erase(dispatch_key(device));
    }
    if (d.get) reinterpret_cast<PFN_vkDestroyDevice>(d.get(device, "vkDestroyDevice"))(device, alloc);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gdpa(VkDevice device, const char *name) {
    if (!strcmp(name, "vkGetDeviceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(gdpa);
    if (!device) return nullptr;
    const auto d = device_for(device);
    if (!d.get) return nullptr;
    if (!strcmp(name, "vkDestroyDevice")) return reinterpret_cast<PFN_vkVoidFunction>(destroy_device);
    if (!strcmp(name, "vkQueueSubmit") && d.submit) return reinterpret_cast<PFN_vkVoidFunction>(submit);
    if (!strcmp(name, "vkQueueSubmit2") && d.submit2) return reinterpret_cast<PFN_vkVoidFunction>(submit2);
    if (!strcmp(name, "vkQueueSubmit2KHR") && d.submit2_khr) return reinterpret_cast<PFN_vkVoidFunction>(submit2_khr);
    if (profile_pipelines() && !strcmp(name, "vkCreateComputePipelines") && d.create_pipelines)
        return reinterpret_cast<PFN_vkVoidFunction>(create_pipelines);
    return d.get(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gipa(VkInstance instance, const char *name) {
    if (!strcmp(name, "vkGetInstanceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(gipa);
    if (!strcmp(name, "vkGetDeviceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(gdpa);
    if (!strcmp(name, "vkCreateInstance")) return reinterpret_cast<PFN_vkVoidFunction>(create_instance);
    if (!instance) return nullptr;
    if (!strcmp(name, "vkDestroyInstance")) return reinterpret_cast<PFN_vkVoidFunction>(destroy_instance);
    if (!strcmp(name, "vkCreateDevice")) return reinterpret_cast<PFN_vkVoidFunction>(create_device);
    if (!strcmp(name, "vkDestroyDevice")) return reinterpret_cast<PFN_vkVoidFunction>(destroy_device);
    if (!strcmp(name, "vkQueueSubmit")) return reinterpret_cast<PFN_vkVoidFunction>(submit);
    if (!strcmp(name, "vkQueueSubmit2")) return reinterpret_cast<PFN_vkVoidFunction>(submit2);
    if (!strcmp(name, "vkQueueSubmit2KHR")) return reinterpret_cast<PFN_vkVoidFunction>(submit2_khr);
    if (profile_pipelines() && !strcmp(name, "vkCreateComputePipelines")) return reinterpret_cast<PFN_vkVoidFunction>(create_pipelines);
    PFN_vkGetInstanceProcAddr next{};
    {
        std::lock_guard<std::mutex> lock(m);
        if (auto it = instances.find(dispatch_key(instance)); it != instances.end()) next = it->second.get;
    }
    return next ? next(instance, name) : nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gpdpa(VkInstance instance, const char *name) {
    PFN_GetPhysicalDeviceProcAddr next{};
    {
        std::lock_guard<std::mutex> lock(m);
        if (auto it = instances.find(dispatch_key(instance)); it != instances.end()) next = it->second.get_physical;
    }
    return next ? next(instance, name) : nullptr;
}
}

extern "C" __declspec(dllexport) VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *version) {
    if (!version || version->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT || version->loaderLayerInterfaceVersion < 2)
        return VK_ERROR_INITIALIZATION_FAILED;
    version->loaderLayerInterfaceVersion = 2;
    version->pfnGetInstanceProcAddr = gipa;
    version->pfnGetDeviceProcAddr = gdpa;
    version->pfnGetPhysicalDeviceProcAddr = gpdpa;
    return VK_SUCCESS;
}

extern "C" __declspec(dllexport) int r10_trace_arm(uint64_t arm) {
    std::lock_guard<std::mutex> lock(m);
    transfer_arm = arm;
    if (arm) observation = Observation{arm};
    return int(devices.size());
}

extern "C" __declspec(dllexport) int r10_trace_wait(uint64_t arm, uint32_t timeout_ms, R10TraceRecord *out) {
    Observation found;
    {
        std::unique_lock<std::mutex> lock(m);
        if (!captured.wait_for(lock, std::chrono::milliseconds(timeout_ms), [arm] {
            return observation.arm == arm && observation.signal != VK_NULL_HANDLE;
        })) return R10_TRACE_NO_SIGNAL;
        found = observation;
    }
    const VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, nullptr, 0, 1, &found.signal, &found.value};
    const auto rc = found.device.wait(found.device.handle, &info, uint64_t(timeout_ms) * 1000000);
    *out = {found.submitted_qpc, qpc(), reinterpret_cast<uintptr_t>(found.queue), found.value};
    return rc == VK_SUCCESS ? R10_TRACE_OK : rc == VK_TIMEOUT ? R10_TRACE_WAIT_TIMEOUT : R10_TRACE_WAIT_ERROR;
}
