#include "r10_catalog_transfer.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>

namespace {
using namespace r10cat;
using Clock = std::chrono::steady_clock;
constexpr size_t MiB = 1024 * 1024;
constexpr std::array<size_t, 4> sizes{8 * MiB, 16 * MiB, 32 * MiB, 52690944};

LONG WINAPI fault(EXCEPTION_POINTERS *ex) {
    std::fprintf(stderr, "r10_transfer_grid_fault,0x%08lx\n", ex->ExceptionRecord->ExceptionCode);
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}
int64_t us(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count();
}
Bytes read_buffered(const fs::path &root, size_t id) {
    Bytes payload(sizes[id]);
    std::ifstream file(path_for(root, id, false), std::ios::binary);
    need(bool(file.read(reinterpret_cast<char *>(payload.data()), payload.size())) && file.peek() == std::char_traits<char>::eof(),
         "transfer calibration source has wrong byte count");
    return payload;
}
Target layout(size_t id, ggml_tensor *span, size_t groups) {
    Target output;
    const auto stride = sizes[id] / groups;
    const auto base = reinterpret_cast<uintptr_t>(span->data);
    need(stride % 4 == 0 && stride * groups == sizes[id], "calibration tensor range is unaligned");
    for (size_t at = 0; at < sizes[id]; at += stride) {
        output[id].push_back({{span->buffer, base + at, stride, at}, span, at});
    }
    return output;
}
}

int main(int argc, char **argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    SetUnhandledExceptionFilter(fault);
    need(argc == 6, "usage: r10_transfer_grid DLLDIR DEVICE PREPARED_OBJECT_DIR REPEATS SEED");
    need(std::getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") && std::getenv("GGML_VK_R10_CONCURRENT_BUFFERS"), "explicit native async opt-ins required");
    ggml_backend_load_all_from_path(argv[1]);
    const auto device = ggml_backend_dev_by_name(argv[2]);
    need(device, "calibration GPU unavailable");
    const fs::path root(argv[3]);
    const auto reps = std::stoul(argv[4]);
    std::mt19937 random(std::stoul(argv[5]));
    need(reps > 0 && reps <= 50, "invalid repeat count");
    Plan p;
    std::array<Bytes, sizes.size()> expected;
    for (size_t id = 0; id < sizes.size(); ++id) {
        p.objects.push_back({id + 1 == sizes.size() ? 'R' : 'K', sizes[id], {}});
        expected[id] = read_buffered(root, id);
    }
    R10StagePool pool(64 * MiB, std::vector<size_t>(sizes.begin(), sizes.end()));
    DeviceTransfer lane(device, p, root, pool.capacity());
    DeviceTransfer::TensorMeta tensor{ggml_init({ggml_tensor_overhead(), nullptr, true}), ggml_free};
    need(bool(tensor), "native transfer grid tensor allocation failed");
    auto *target = ggml_new_tensor_1d(tensor.get(), GGML_TYPE_I8, sizes.back());
    DeviceTransfer::Buffer target_storage{ggml_backend_alloc_ctx_tensors(tensor.get(), lane.compute_marker_backend()), ggml_backend_buffer_free};
    need(target && target_storage && !ggml_backend_buffer_is_host(target_storage.get()), "calibration target is not device memory");
    std::array<size_t, sizes.size()> order{0, 1, 2, 3};
    uint64_t last_fence = 0;
    std::cout << "repeat,order,tier,bytes,host_to_submit_us,disk_us,submit_to_visible_us,end_to_end_us,device_scatter_us,process_read_bytes,exact\n";
    for (int rep = -1; rep < int(reps); ++rep) {
        std::shuffle(order.begin(), order.end(), random);
        for (size_t index = 0; index < order.size(); ++index) {
            const auto id = order[index];
            const auto bytes = sizes[id];
            need(pool.reserve(0, {id}), "isolated calibration object did not fit");
            int64_t submit = -1, end = -1, disk = 0;
            uint64_t ticket = 0;
            IO_COUNTERS before{}, after{};
            need(GetProcessIoCounters(GetCurrentProcess(), &before), "cannot sample transfer read counters");
            const auto start = Clock::now();
            auto observe = [&](const DeviceTransfer::Step &step) {
                if (step.ssd_bytes) {
                    need(step.ssd_bytes == bytes, "unbuffered read byte mismatch");
                    disk = step.ssd_observation_us;
                }
                if (step.submitted) {
                    submit = us(Clock::now());
                    need(step.submitted > last_fence && !ticket, "transfer ticket is not monotonic");
                    pool.issued(id, step.submitted);
                    ticket = last_fence = step.submitted;
                }
                if (step.completed) {
                    end = us(Clock::now());
                    need(ticket && step.completed == ticket, "transfer completed with wrong identity");
                    pool.landed(id, step.completed);
                }
            };
            observe(lane.read(id, nullptr, pool));
            while (lane.busy()) {
                observe(lane.poll(pool));
                need(Clock::now() - start < std::chrono::seconds(5), "calibration transfer timed out");
                if (lane.busy()) std::this_thread::yield();
            }
            need(submit >= us(start) && end >= submit && pool.consumable(0) && (id < 3 ? disk == 0 : disk > 0),
                 "incomplete byte or timing evidence");
            need(GetProcessIoCounters(GetCurrentProcess(), &after) && after.ReadTransferCount >= before.ReadTransferCount,
                 "invalid process I/O counter delta");
            const auto read_counted = after.ReadTransferCount - before.ReadTransferCount;
            need(id < 3 || read_counted >= bytes, "OS did not account for the unbuffered object read");
            // Make 16 separate target spans in this device-only probe. Actual
            // recurrent and fragmented live scatter layouts are timed separately.
            const auto spans = layout(id, target, 16);
            const auto scatter = Clock::now();
            need(lane.device_copy(id, spans, pool, false) == bytes, "device scatter byte count changed");
            const auto copied = us(Clock::now()) - us(scatter);
            Bytes result(bytes);
            ggml_backend_tensor_get(target, result.data(), 0, result.size());
            need(result == expected[id], "isolated transfer and scatter did not reproduce host bytes");
            pool.release(0);
            if (rep >= 0) {
                std::cout << rep << ',' << index << ',' << (id == 3 ? "SSD" : "RAM") << ',' << bytes << ','
                          << submit - us(start) << ',' << disk << ',' << end - submit << ',' << end - us(start)
                          << ',' << copied << ',' << read_counted << ",1\n" << std::flush;
            }
        }
    }
    return 0;
} catch (const std::exception &error) {
    std::cerr << "r10 transfer grid: " << error.what() << '\n';
    return 1;
}
