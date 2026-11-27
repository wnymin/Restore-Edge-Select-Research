#include "r10_catalog_source.h"

#include <iostream>
#include <thread>

// Qualify the polling and exact delivery side of the adapter in isolation.
// This is not a bandwidth benchmark: the buffered equality check is deliberate.
int main(int argc, const char **argv) try {
    using namespace r10cat;
    need(argc >= 2, "usage: r10_catalog_disk_probe CHECKPOINT [CHECKPOINT ...]");
    DiskRead io(52690944);
    std::cout << "file,bytes,incomplete_polls,completion_observed_us,content_equal\n";
    for (int i = 1; i < argc; ++i) {
        const fs::path file = argv[i];
        const auto size = fs::file_size(file);
        io.start(file, size);
        bool denied = false;
        try { (void) io.payload(size); } catch (const std::runtime_error &) { denied = true; }
        need(denied, "SSD data was visible before any verified completion");
        denied = false;
        try { io.start(file, size); } catch (const std::runtime_error &) { denied = true; }
        need(denied, "single disk lane accepted a second pending object");
        const auto until = DiskRead::Clock::now() + std::chrono::seconds(10);
        size_t polls = 0;
        std::optional<DiskRead::Completion> finish;
        while (!(finish = io.poll())) {
            ++polls;
            need(DiskRead::Clock::now() < until, "SSD poll did not complete in 10 seconds");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        need(finish->bytes == size, "wrong completed disk byte count");
        std::ifstream buffered(file, std::ios::binary);
        Bytes data(size);
        need(bool(buffered.read(reinterpret_cast<char *>(data.data()), size)) && buffered.peek() == std::char_traits<char>::eof() &&
             std::equal(data.begin(), data.end(), io.payload(size)), "overlapped bytes differ from buffered reference");
        std::cout << file.filename().string() << ',' << finish->bytes << ',' << polls << ',' << finish->observed_us << ",1\n";
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << "r10 catalogue disk probe: " << e.what() << '\n';
    return 1;
}
