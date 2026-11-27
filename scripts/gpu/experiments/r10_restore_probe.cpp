#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<uint8_t>;
using Tokens = std::vector<llama_token>;
constexpr int prompt_length = 2048;
constexpr int positions = 8;

void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void append(llama_context *ctx, llama_token *tokens, int count, int batch) {
    for (int offset = 0; offset < count; offset += batch) {
        check(llama_decode(ctx, llama_batch_get_one(tokens + offset, std::min(batch, count - offset))) == 0,
              "decode failed");
    }
    llama_synchronize(ctx);
}

Bytes save(llama_context *ctx, llama_state_seq_flags flags) {
    Bytes state(llama_state_seq_get_size_ext(ctx, 0, flags));
    check(!state.empty() && llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, flags) == state.size(),
          "state capture failed");
    return state;
}

void write_aligned(const std::filesystem::path &path, const Bytes &bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    const Bytes padding((4096 - bytes.size() % 4096) % 4096, 0);
    if (!padding.empty()) out.write(reinterpret_cast<const char *>(padding.data()), padding.size());
    out.close();
    check(bool(out), "cannot stage padded recurrent file");
}

void read_aligned(const std::filesystem::path &path, Bytes &destination, size_t prefix_size) {
    const size_t meaningful = destination.size() - prefix_size;
    const size_t read_size = (meaningful + 4095) / 4096 * 4096;
    auto *raw = static_cast<uint8_t *>(VirtualAlloc(nullptr, read_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    check(raw, "cannot allocate aligned read buffer");
    struct Release { uint8_t *raw; ~Release() { VirtualFree(raw, 0, MEM_RELEASE); } } release{raw};
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_NO_BUFFERING, nullptr);
    check(file != INVALID_HANDLE_VALUE, "cannot open recurrent file with page cache bypass");
    DWORD got = 0;
    const BOOL read = ReadFile(file, raw, DWORD(read_size), &got, nullptr);
    CloseHandle(file);
    check(read && got == read_size, "short aligned recurrent read");
    std::copy_n(raw, meaningful, destination.begin() + prefix_size);
}

std::vector<float> continuations(llama_context *ctx, int vocab, Tokens &history, bool choose_history) {
    std::vector<float> output;
    output.reserve(positions * vocab);
    for (int i = 0; i < positions; ++i) {
        const auto *logits = llama_get_logits_ith(ctx, -1);
        check(logits && std::all_of(logits, logits + vocab, [](float v) { return std::isfinite(v); }),
              "missing or nonfinite logits");
        output.insert(output.end(), logits, logits + vocab);
        if (i + 1 == positions) break;
        if (choose_history) history.push_back(llama_token(std::max_element(logits, logits + vocab) - logits));
        append(ctx, &history.at(i), 1, 1);
    }
    return output;
}

struct Comparison { int top_matches; double max_error; };

Comparison compare(const std::vector<float> &base, const std::vector<float> &other, int vocab) {
    Comparison result{0, 0.0};
    for (int i = 0; i < positions; ++i) {
        auto a = base.begin() + i * vocab, b = other.begin() + i * vocab;
        if (std::max_element(a, a + vocab) - a == std::max_element(b, b + vocab) - b) ++result.top_matches;
        for (int j = 0; j < vocab; ++j) {
            result.max_error = std::max(result.max_error, std::abs(double(a[j]) - double(b[j])));
        }
    }
    return result;
}
}

int main(int argc, char **argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    check(argc == 6, "usage: r10_restore_probe MODEL BACKENDS DEVICE SCRATCH REPEATS");
    const auto repeats = std::stoi(argv[5]);
    const std::filesystem::path recurrent_path = std::string(argv[4]) + ".recurrent";
    check(repeats > 0 && !std::filesystem::exists(argv[4]) && !std::filesystem::exists(recurrent_path),
          "require positive repeats and new scratch paths");
    struct ScratchFile {
        std::filesystem::path path;
        ~ScratchFile() { std::error_code ignored; std::filesystem::remove(path, ignored); }
    } scratch{argv[4]}, recurrent_scratch{recurrent_path};
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name(argv[3]), nullptr};
    check(devices[0], "requested GPU backend not available");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    check(bool(model), "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_seq_max = 2;
    cp.kv_unified = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> owned(llama_init_from_model(model.get(), cp), llama_free);
    check(bool(owned), "context init failed");
    auto *ctx = owned.get();
    const auto mem = llama_get_memory(ctx);
    const auto vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::mt19937 tokens(10131), order(181719);
    std::cout << "rep,depth,arm,full_bytes,partial_bytes,file_bytes,load_ms,top_matches_cut,max_error_cut,"
                 "top_matches_full,max_error_full,state_bytes_equal\n" << std::setprecision(10);
    for (int rep = -1; rep < repeats; ++rep) {
        Tokens prompt(prompt_length);
        for (auto &token : prompt) token = 1000 + tokens() % 14000;
        Tokens history;
        llama_memory_clear(mem, false);
        append(ctx, prompt.data(), prompt.size(), 512);
        const auto full = continuations(ctx, vocab, history, true);
        for (int depth : {512, 1024, 1536}) {
            llama_memory_clear(mem, false);
            append(ctx, prompt.data(), depth, 256);
            const auto data = save(ctx, LLAMA_STATE_SEQ_FLAGS_NONE);
            const auto partial = save(ctx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            // Pinned sequence format prefixes both encodings with magic and source sequence ID.
            constexpr size_t header_size = sizeof(uint32_t) + sizeof(llama_seq_id);
            check(partial.size() > header_size && partial.size() < data.size() &&
                  std::equal(partial.begin(), partial.begin() + header_size, data.begin()) &&
                  std::equal(partial.rbegin(), partial.rend() - header_size, data.rbegin()),
                  "recurrent serialization disagrees with full-state encoding");
            const size_t attention_size = data.size() - partial.size() + header_size;
            Bytes attention(data.begin(), data.begin() + attention_size);
            write_aligned(recurrent_path, Bytes(partial.begin() + header_size, partial.end()));
            const auto written = llama_state_seq_save_file(ctx, argv[4], 0, prompt.data(), depth);
            check(written > 0, "file save failed");
            append(ctx, prompt.data() + depth, prompt.size() - depth, 256);
            const auto cut = continuations(ctx, vocab, history, false);
            const auto numeric_control = compare(full, cut, vocab);
            std::cout << rep << ',' << depth << ",recompute-256," << data.size() << ',' << partial.size() << ','
                      << written << ",0," << positions << ",0," << numeric_control.top_matches << ','
                      << numeric_control.max_error << ",\n";
            std::vector<std::string> arms = {"ram-direct", "ram-handoff", "file-handoff", "split-host-handoff"};
            std::shuffle(arms.begin(), arms.end(), order);
            Bytes reconstructed(data.size());
            for (const auto &arm : arms) {
                llama_memory_clear(mem, false);
                const llama_seq_id seq = arm == "ram-direct" ? 0 : 1;
                Tokens loaded(depth);
                size_t count = 0;
                const auto start = Clock::now();
                if (arm == "file-handoff") {
                    check(llama_state_seq_load_file(ctx, argv[4], seq, loaded.data(), loaded.size(), &count) == written,
                          "file restore failed");
                    check(count == loaded.size() && std::equal(loaded.begin(), loaded.end(), prompt.begin()),
                          "file restore returned wrong token history");
                } else if (arm == "split-host-handoff") {
                    std::copy(attention.begin(), attention.end(), reconstructed.begin());
                    read_aligned(recurrent_path, reconstructed, attention_size);
                    check(llama_state_seq_set_data(ctx, reconstructed.data(), reconstructed.size(), seq) == reconstructed.size(),
                          "reconstructed state restore failed");
                } else {
                    check(llama_state_seq_set_data(ctx, data.data(), data.size(), seq) == data.size(), "RAM restore failed");
                }
                llama_synchronize(ctx);
                check(llama_memory_seq_pos_max(mem, seq) == depth - 1, "staged position mismatch");
                if (seq != 0) {
                    llama_memory_seq_cp(mem, seq, 0, -1, -1);
                    check(llama_memory_seq_rm(mem, seq, -1, -1), "temporary sequence release failed");
                    check(llama_memory_seq_pos_max(mem, seq) < 0, "temporary sequence still visible");
                }
                check(llama_memory_seq_pos_max(mem, 0) == depth - 1, "consumed position mismatch");
                const auto load_ms = elapsed_ms(start);
                const bool equal = save(ctx, LLAMA_STATE_SEQ_FLAGS_NONE) == data;
                append(ctx, prompt.data() + depth, prompt.size() - depth, 256);
                const auto restored = continuations(ctx, vocab, history, false);
                const auto a = compare(cut, restored, vocab), b = compare(full, restored, vocab);
                std::cout << rep << ',' << depth << ',' << arm << ',' << data.size() << ',' << partial.size() << ','
                          << written << ',' << load_ms << ',' << a.top_matches << ',' << a.max_error << ','
                          << b.top_matches << ',' << b.max_error << ',' << equal << std::endl;
            }
        }
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
