#include "r10_catalog_ledger.h"
#include "r10_catalog_transfer.h"
#include "r10_vk_trace_abi.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <sstream>
#include <psapi.h>

namespace {
using namespace r10cat;
using Clock = std::chrono::steady_clock;
constexpr size_t chunk = 256;
const char *volatile phase = "startup";

LONG WINAPI fault(EXCEPTION_POINTERS *ex) {
    std::fprintf(stderr, "r10_catalog_worker_fault,0x%08lx,phase=%s\n", ex->ExceptionRecord->ExceptionCode, phase);
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}

void resource_sample(const char *where, ggml_backend_dev_t device) {
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(device, &free, &total);
    PROCESS_MEMORY_COUNTERS_EX mem{};
    need(K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&mem), sizeof(mem)),
         "process resource sample unavailable");
    std::cerr << "r10_resources," << where << ',' << free << ',' << total << ',' << mem.WorkingSetSize << ','
              << mem.PeakWorkingSetSize << ',' << mem.PrivateUsage << '\n';
}

int64_t elapsed_us(Clock::time_point before) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - before).count();
}

uint64_t host_qpc() {
    LARGE_INTEGER value{};
    need(QueryPerformanceCounter(&value), "missing host QPC clock");
    return uint64_t(value.QuadPart);
}

struct ComputeMarker {
    explicit ComputeMarker(ggml_backend_dev_t device, const fs::path &library) :
        layer(LoadLibraryW(fs::absolute(library).c_str())), marker(ggml_backend_event_new(device), ggml_backend_event_free) {
        need(layer && marker, "native compute witness unavailable");
        load(arm, "r10_trace_arm");
        load(wait, "r10_trace_wait");
    }
    ~ComputeMarker() { if (layer) FreeLibrary(layer); }
    ComputeMarker(const ComputeMarker &) = delete;
    ComputeMarker &operator=(const ComputeMarker &) = delete;

    void record(ggml_backend_t backend) {
        need(arm(++tag) > 0, "could not arm real compute signal");
        ggml_backend_event_record(marker.get(), backend);
        arm(0);
    }
    bool ready() const {
        R10TraceRecord result{};
        const int state = wait(tag, 0, &result);
        need(result.queue && result.timeline_value && (state == R10_TRACE_OK || state == R10_TRACE_WAIT_TIMEOUT),
             "cannot observe native compute completion");
        return state == R10_TRACE_OK;
    }

private:
    template<typename F> void load(F &dst, const char *name) {
        const auto proc = GetProcAddress(layer, name);
        static_assert(sizeof(dst) == sizeof(proc));
        std::memcpy(&dst, &proc, sizeof(proc));
        need(dst != nullptr, name);
    }
    HMODULE layer;
    std::unique_ptr<ggml_backend_event, decltype(&ggml_backend_event_free)> marker;
    R10TraceArm arm{};
    R10TraceWait wait{};
    uint64_t tag = 0;
};

struct Worker {
    Worker(Plan plan, llama_context *ctx, ggml_backend_dev_t device, size_t cap, fs::path root, const fs::path &layer,
           std::chrono::milliseconds ssd_hold) :
        plan_(std::move(plan)), ctx_(ctx), root_(std::move(root)), ledger_(cap, sizes(plan_)),
        device_(device, plan_, root_, cap, ssd_hold), compute_(device, layer) {}

    void configure_service(const llama_vocab *vocab, size_t short_limit, size_t long_limit) {
        vocab_ = vocab;
        vocabulary_ = size_t(llama_vocab_n_tokens(vocab));
        short_limit_ = short_limit;
        long_limit_ = long_limit;
        need(vocabulary_ > 0, "service requires a nonempty vocabulary");
        for (size_t ri = 0; ri < plan_.requests.size(); ++ri) {
            const auto &r = plan_.requests[ri];
            // A checkpoint holds state, not the logits of its final token.
            // Do not silently change the legal actions or emulate the first token.
            need(r.tokens.size() + limit(ri) - 1 <= plan_.context && r.options.rbegin()->first < r.tokens.size(),
                 "service requires context headroom and one prompt token after every candidate");
        }
    }

    static std::vector<size_t> sizes(const Plan &p) {
        std::vector<size_t> result;
        for (const auto &o : p.objects) result.push_back(o.bytes);
        return result;
    }
    const Candidate &candidate(size_t req, size_t d) {
        const auto &c = plan_.requests.at(req).options.at(d);
        need(bindings_.emplace(req, d).first->second == d, "request boundary was changed after publication");
        return c;
    }
    void event(const char *type, size_t req, size_t identity, size_t bytes, uint64_t ticket = 0, int64_t interval = 0) const {
        if (quiet_) return;
        const auto t = elapsed_us(clock_);
        std::cout << type << ',' << t << ',' << req << ',' << identity << ',' << bytes << ',' << ledger_.pool().charged()
                  << ',' << ticket << ',' << interval << '\n';
    }
    void retarget(size_t req, const Candidate &c) {
        need(req == next_ && !computing_ && !head_, "overlapping native target ownership");
        llama_synchronize(ctx_);
        llama_memory_clear(llama_get_memory(ctx_), false);
        Target target;
        if (c.depth) {
            auto disk = load_meta(path_for(root_, c.deps.back(), true), plan_, c);
            target = bind_target(ctx_, disk, device_.device_type());
        }
        std::set<size_t> keep;
        for (auto id : ledger_.aliases()) {
            const auto it = target.find(id);
            if (it != target.end() && plan_.objects.at(id).kind == 'K' && same_native_bytes(current_.at(id), it->second)) {
                keep.insert(id);
                event("carry-active-kv", req, id, plan_.objects.at(id).bytes);
            }
        }
        for (auto id : ledger_.before_switch(keep)) materialize(req, id, "preserve-old-active");
        if (!c.depth) llama_memory_clear(llama_get_memory(ctx_), true);
        current_ = std::move(target);
        ledger_.bind_head(req, c.deps, keep);
        head_ = true;
        event("native-target", req, c.depth, 0);
    }
    void materialize(size_t req, size_t id, const char *type) {
        const auto start = Clock::now();
        const auto bytes = device_.device_copy(id, current_, ledger_.pool(), true);
        const auto duration = elapsed_us(start);
        ledger_.materialized(id);
        event(type, req, id, bytes, 0, duration);
    }
    void observe_transfer(DeviceTransfer::Step outcome) {
        if (outcome.ssd_bytes) event("disk-complete", reading_request_, reading_id_, outcome.ssd_bytes, 0, outcome.ssd_observation_us);
        if (outcome.soft_hold_us) event("disk-soft-hold", reading_request_, reading_id_, plan_.objects.at(reading_id_).bytes, 0, outcome.soft_hold_us);
        if (outcome.submitted) {
            ledger_.submitted(outcome.submitted);
            event("upload-submit", reading_request_, reading_id_, plan_.objects.at(reading_id_).bytes, outcome.submitted);
        }
        if (outcome.completed) {
            ledger_.landed(outcome.completed);
            event("upload-ready", reading_request_, reading_id_, plan_.objects.at(reading_id_).bytes, outcome.completed);
        }
    }
    void submit_chunk() {
        auto &tokens = plan_.requests.at(next_).tokens;
        const auto n = std::min(chunk, tokens.size() - pos_);
        if (chunk_events_) event("chunk-submit-begin", next_, pos_ + n, 0, host_qpc());
        const auto decode_start = chunk_events_ ? Clock::now() : Clock::time_point{};
        need(n && llama_decode(ctx_, llama_batch_get_one(tokens.data() + pos_, int(n))) == 0, "native prompt chunk failed");
        const auto decode_duration = chunk_events_ ? elapsed_us(decode_start) : 0;
        pos_ += n;
        compute_.record(device_.compute_marker_backend());
        if (chunk_events_) {
            last_chunk_pending_ = {};
            event("chunk-marker", next_, pos_, 0, host_qpc(), decode_duration);
        }
    }

    // Keep raw reference bytes for a small correctness fixture. This is
    // deliberately different from the timing path and runs before the epoch.
    void qualification(size_t vocabulary) {
        need(vocabulary && plan_.requests.size() <= 12, "state qualification is for fixtures of at most 12 requests");
        phase = "independent full recompute references";
        vocabulary_ = vocabulary;
        for (size_t ri = 0; ri < plan_.requests.size(); ++ri) {
            auto &row = plan_.requests[ri];
            llama_synchronize(ctx_);
            llama_memory_clear(llama_get_memory(ctx_), true);
            for (size_t pos = 0; pos < row.tokens.size(); pos += chunk) {
                const auto n = std::min(chunk, row.tokens.size() - pos);
                need(llama_decode(ctx_, llama_batch_get_one(row.tokens.data() + pos, int(n))) == 0, "reference prompt chunk failed");
            }
            const auto *logits = llama_get_logits_ith(ctx_, -1);
            need(logits && std::all_of(logits, logits + vocabulary_, [](float x) { return std::isfinite(x); }), "nonfinite reference logits");
            references_.push_back({snapshot(), std::vector<float>(logits, logits + vocabulary_), {}, false});
            if (short_limit_) {
                auto &ref = references_.back();
                while (ref.output.size() < limit(ri)) {
                    auto token = greedy();
                    if (llama_vocab_is_eog(vocab_, token)) {
                        ref.eog = true;
                        break;
                    }
                    ref.output.push_back(token);
                    if (ref.output.size() == limit(ri)) break;
                    need(llama_decode(ctx_, llama_batch_get_one(&token, 1)) == 0, "reference decode token failed");
                    llama_synchronize(ctx_);
                }
            }
        }
        llama_synchronize(ctx_);
        llama_memory_clear(llama_get_memory(ctx_), true);
    }

    void warmup(bool grid) {
        phase = "untimed uniform warmup";
        Tokens tokens(plan_.context);
        for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = llama_token(301 + i * 13);
        auto sequence = [&](size_t end) {
            if (end > tokens.size()) return;
            llama_memory_clear(llama_get_memory(ctx_), true);
            for (size_t pos = 0; pos < end; pos += chunk) {
                const auto n = std::min(chunk, end - pos);
                need(llama_decode(ctx_, llama_batch_get_one(tokens.data() + pos, int(n))) == 0, "common warmup failed");
                llama_synchronize(ctx_);
            }
        };
        if (grid) {
            // Fixed synthetic token histories, chosen on calibration only. No
            // input/candidate state from any timed block is read by warmup.
            constexpr std::array<size_t, 18> shapes{1, 16, 64, 128, 192, 256, 304, 384, 512,
                                                   520, 576, 608, 704, 1024, 1088, 1536, 2048, 2560};
            const auto before = Clock::now();
            for (auto n : shapes) sequence(n);
            std::cerr << "r10_warmup_grid_us," << elapsed_us(before) << '\n';
        } else sequence(chunk);
        llama_memory_clear(llama_get_memory(ctx_), true);
    }

    void report_allocation() const {
        std::cout << "!allocated " << plan_.requests.size() << ' ' << device_.arena_bytes() << ' '
                  << device_.resident_ram_bytes() << ' ' << device_.pinned_bytes() << std::endl;
    }

    void run(bool quiet, bool trace_chunks) {
        quiet_ = quiet;
        chunk_events_ = trace_chunks;
        if (chunk_events_) {
            LARGE_INTEGER hz{};
            need(QueryPerformanceFrequency(&hz), "missing host QPC frequency");
            std::cerr << "r10_chunk_clock," << hz.QuadPart << '\n';
        }
        clock_ = Clock::now();
        std::cout << "event,native_us,request,identity,bytes,charged_bytes,ticket,interval_us\n";
        std::cout << "!open " << plan_.requests.size() << ' ' << device_.arena_bytes() << ' '
                  << device_.resident_ram_bytes() << ' ' << device_.pinned_bytes() << std::endl;
        for (std::string line; std::getline(std::cin, line);) {
            std::istringstream in(line);
            std::string command, more;
            need(bool(in >> command), "empty native command");
            phase = "execution pipe";
            if (command == "reserve") {
                size_t req, d, n;
                need(bool(in >> req >> d >> n) && n <= 3 && req > next_, "invalid reservation request");
                const auto &c = candidate(req, d);
                std::set<size_t> loans;
                for (size_t j = 0, id; j < n; ++j) need(bool(in >> id) && loans.insert(id).second, "duplicate or missing active loan");
                need(!(in >> more), "unexpected reservation field");
                ledger_.reserve(req, c.deps, loans);
                event("reserved", req, d, 0);
                std::cout << "!ok" << std::endl;
            } else if (command == "read") {
                size_t req, d, id;
                std::string use;
                need(bool(in >> req >> d >> id >> use) && !(in >> more) && !device_.busy() &&
                     ((use == "demand" && req == next_ && !computing_) || (use == "prefetch" && req > next_)),
                     "invalid native read request");
                const auto &c = candidate(req, d);
                need(std::find(c.deps.begin(), c.deps.end(), id) != c.deps.end(), "read object is outside selected candidate");
                if (!head_ && ledger_.pool().get(id).state == R10StagePool::State::absent) retarget(req, c);
                const auto op = ledger_.dispatch(req, id, use == "demand");
                reading_request_ = req;
                reading_id_ = id;
                event(op.direct ? "read-direct" : "read-staged", req, id, plan_.objects[id].bytes);
                observe_transfer(device_.read(id, op.direct ? &current_ : nullptr, ledger_.pool()));
                std::cout << "!ok" << std::endl;
            } else if (command == "compute") {
                size_t req, d, release, shared;
                need(bool(in >> req >> d >> release >> shared) && !(in >> more) && req == next_ && !computing_ && release <= 1 && shared <= 1,
                     "invalid native compute request");
                const auto &c = candidate(req, d);
                if (!head_) retarget(req, c);
                for (auto id : c.deps) if (ledger_.source(id) == PayloadLedger::Source::staged) {
                    const auto start = Clock::now();
                    const auto bytes = device_.device_copy(id, current_, ledger_.pool(), false);
                    const auto duration = elapsed_us(start);
                    ledger_.scattered(id);
                    event("native-scatter", req, id, bytes, 0, duration);
                }
                const bool has_suffix = d < plan_.requests.at(req).tokens.size();
                const bool shared_locally = has_suffix && !c.deps.empty() && ledger_.pool().get(c.deps.back()).refs > release;
                need(shared_locally == bool(shared), "controller and physical pool disagree about shared checkpoint");
                if (ledger_.recurrent_must_preserve(release, has_suffix)) materialize(req, c.deps.back(), "preserve-recurrent");
                ledger_.begin_compute(req, release, has_suffix);
                pos_ = d;
                if (has_suffix) {
                    submit_chunk();
                    computing_ = true;
                    event("suffix-submitted", req, d, 0);
                } else {
                    verify_prefill(req);
                    finish(req);
                }
                std::cout << "!ok " << int(!has_suffix) << std::endl;
            } else if (command == "poll") {
                need(!(in >> more), "unexpected poll field");
                const auto t = device_.poll(ledger_.pool());
                observe_transfer(t);
                int64_t done = -1;
                if (computing_) {
                    // The previous unsuccessful query began before the signal.
                    const auto query = chunk_events_ ? Clock::now() : Clock::time_point{};
                    if (compute_.ready()) {
                        if (chunk_events_ && !decoding_) event("chunk-ready", next_, pos_, 0, 0,
                            last_chunk_pending_ == Clock::time_point{} ? 0 : elapsed_us(last_chunk_pending_));
                        if (pos_ < plan_.requests.at(next_).tokens.size()) submit_chunk();
                        else if (short_limit_) {
                            if (!decoding_) {
                                verify_prefill(next_);
                                decoding_ = true;
                                service_time("prefill", next_);
                                event("service-prefill-done", next_, pos_, 0);
                            }
                            if (advance_service(next_)) {
                                done = int64_t(next_);
                                finish(next_);
                            }
                        }
                        else {
                            done = int64_t(next_);
                            verify_prefill(next_);
                            finish(next_);
                        }
                    } else if (chunk_events_) {
                        last_chunk_pending_ = query;
                    }
                }
                std::cout << "!poll " << t.completed << ' ' << done << std::endl;
            } else if (command == "finish") {
                need(!(in >> more) && next_ == plan_.requests.size() && ledger_.drained() && !device_.busy(), "native FIFO stopped before draining");
                std::cout << "!done" << std::endl;
                return;
            } else need(false, "unknown native pipe command");
        }
        need(false, "native worker pipe closed before finish");
    }

    void verify_prefill(size_t req) {
        if (!references_.empty()) {
            const auto actual = snapshot();
            const auto &expected = references_.at(req);
            if (actual != expected.state) {
                const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.state.begin(), expected.state.end());
                throw std::runtime_error("online state mismatch for request " + std::to_string(req) +
                                         " at byte " + std::to_string(mismatch.first - actual.begin()) +
                                         " of " + std::to_string(actual.size()) + "/" + std::to_string(expected.state.size()));
            }
            event("exact-state", req, bindings_.at(req), actual.size());
            if (bindings_.at(req) < pos_) {
                const auto *l = llama_get_logits_ith(ctx_, -1);
                need(l && std::memcmp(l, expected.logits.data(), vocabulary_ * sizeof(float)) == 0, "final online logits mismatch");
                event("exact-final-logits", req, bindings_.at(req), vocabulary_ * sizeof(float));
            }
        }
    }

    void finish(size_t req) {
        ledger_.computed(req);
        event("compute-done", req, pos_, 0);
        head_ = computing_ = decoding_ = false;
        ++next_;
    }

private:
    size_t limit(size_t request) const { return long_limit_ && request % 4 == 0 ? long_limit_ : short_limit_; }

    llama_token greedy() const {
        const auto *logits = llama_get_logits_ith(ctx_, -1);
        need(logits && std::all_of(logits, logits + vocabulary_, [](float x) { return std::isfinite(x); }),
             "missing or nonfinite generation logits");
        return llama_token(std::max_element(logits, logits + vocabulary_) - logits);
    }

    void service_time(const char *type, size_t req) const {
        std::cout << '~' << type << ' ' << req << ' ' << elapsed_us(clock_) << '\n';
    }

    std::string piece_hex(llama_token token) const {
        std::array<char, 64> buffer{};
        auto n = llama_token_to_piece(vocab_, token, buffer.data(), int32_t(buffer.size()), 0, false);
        std::string extra;
        const char *p = buffer.data();
        if (n < 0) {
            need(n > -32768, "invalid token piece size");
            extra.resize(size_t(-n));
            n = llama_token_to_piece(vocab_, token, extra.data(), int32_t(extra.size()), 0, false);
            need(n >= 0 && size_t(n) <= extra.size(), "could not read the complete token piece");
            p = extra.data();
        } else need(size_t(n) <= buffer.size(), "token piece exceeds small buffer");
        constexpr char hex[] = "0123456789abcdef";
        std::string text;
        for (int32_t i = 0; i < n; ++i) {
            const auto b = static_cast<unsigned char>(p[i]);
            text.push_back(hex[b >> 4]);
            text.push_back(hex[b & 15]);
        }
        return text.empty() ? "-" : text;
    }

    bool advance_service(size_t req) {
        auto token = greedy();
        const bool eog = llama_vocab_is_eog(vocab_, token);
        if (!eog) {
            output_.push_back(token);
            event("service-token", req, output_.size(), 0);
            const auto piece = piece_hex(token);
            std::cout << "~token " << req << ' ' << output_.size() << ' ' << token << ' ' << piece << ' '
                      << elapsed_us(clock_) << std::endl;
        }
        if (eog || output_.size() == limit(req)) {
            if (!references_.empty()) {
                const auto &ref = references_.at(req);
                need(ref.output == output_ && ref.eog == eog, "generated tokens differ from reference recompute");
                event("exact-service-tokens", req, output_.size(), 0);
            }
            std::cout << "~end " << req << ' ' << int(eog) << ' ' << elapsed_us(clock_) << '\n';
            output_.clear();
            return true;
        }
        const auto submitted = Clock::now();
        need(llama_decode(ctx_, llama_batch_get_one(&token, 1)) == 0, "generation token failed");
        compute_.record(device_.compute_marker_backend());
        event("service-decode-marker", req, output_.size(), 0, 0, elapsed_us(submitted));
        return false;
    }

    Bytes snapshot() const {
        Bytes b(llama_state_seq_get_size_ext(ctx_, 0, LLAMA_STATE_SEQ_FLAGS_NONE));
        need(!b.empty() && llama_state_seq_get_data_ext(ctx_, b.data(), b.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) == b.size(),
             "could not snapshot complete native sequence state");
        return b;
    }
    struct Reference { Bytes state; std::vector<float> logits; Tokens output; bool eog; };
    Plan plan_;
    llama_context *ctx_;
    fs::path root_;
    PayloadLedger ledger_;
    DeviceTransfer device_;
    ComputeMarker compute_;
    Target current_;
    std::map<size_t, size_t> bindings_;
    Clock::time_point clock_{}, last_chunk_pending_{};
    size_t next_ = 0, pos_ = 0, reading_request_ = 0, reading_id_ = 0;
    bool head_ = false, computing_ = false, quiet_ = false, chunk_events_ = false, decoding_ = false;
    size_t vocabulary_ = 0;
    size_t short_limit_ = 0, long_limit_ = 0;
    const llama_vocab *vocab_ = nullptr;
    Tokens output_;
    std::vector<Reference> references_;
};
}

int main(int argc, char **argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    SetUnhandledExceptionFilter(fault);
    need(argc >= 8 && argc <= 17,
         "usage: r10_catalog_worker MODEL DLLDIR DEVICE PLAN CATALOGUE STAGE_BYTES OBSERVER_DLL [--verify] [--warmup] [--warmup-grid] [--resources] [--quiet-events] [--allocate-only] [--chunk-events] [--ssd-hold-ms=N] [--service=SHORT:LARGE_OR_ZERO]");
    std::set<std::string> flags;
    unsigned long ssd_hold_ms = 0;
    size_t short_limit = 0, long_limit = 0;
    for (int i = 8; i < argc; ++i) {
        const std::string f = argv[i];
        if (f.rfind("--service=", 0) == 0) {
            size_t short_chars = 0, long_chars = 0;
            const auto colon = f.find(':', 10);
            need(!short_limit && colon != std::string::npos, "duplicate or incomplete service profile");
            short_limit = std::stoul(f.substr(10, colon - 10), &short_chars);
            long_limit = std::stoul(f.substr(colon + 1), &long_chars);
            need(short_chars == colon - 10 && long_chars == f.size() - colon - 1 && short_limit > 0 && short_limit <= 256 &&
                 (!long_limit || (long_limit > short_limit && long_limit <= 256)), "invalid service output limits");
            continue;
        }
        if (f.rfind("--ssd-hold-ms=", 0) == 0) {
            size_t used = 0;
            need(!ssd_hold_ms, "duplicate soft hold option");
            ssd_hold_ms = std::stoul(f.substr(14), &used);
            need(used == f.size() - 14 && ssd_hold_ms > 0 && ssd_hold_ms <= 1000, "invalid SSD soft hold");
            continue;
        }
        need((f == "--verify" || f == "--warmup" || f == "--warmup-grid" || f == "--resources" || f == "--quiet-events" || f == "--allocate-only" || f == "--chunk-events") &&
             flags.insert(f).second, "invalid or duplicate worker option");
    }
    need(!flags.count("--verify") || !flags.count("--quiet-events"), "state qualification must emit evidence");
    need(!flags.count("--warmup-grid") || flags.count("--warmup"), "warmup grid requires common warmup");
    need(!flags.count("--chunk-events") || (!flags.count("--quiet-events") && !flags.count("--allocate-only")),
         "chunk diagnostics require an executed block with events");
    need(!ssd_hold_ms || (!flags.count("--quiet-events") && !flags.count("--allocate-only")), "soft hold requires timed event rows");
    need(!short_limit || (!flags.count("--quiet-events") && !flags.count("--allocate-only") && !ssd_hold_ms),
         "service requires timed events and native storage");
    need(std::getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") && std::getenv("GGML_VK_R10_CONCURRENT_BUFFERS"), "async native Vulkan opt-ins required");
    auto plan = read_plan(argv[4]);
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;
    ggml_backend_dev_t devices[]{ggml_backend_dev_by_name(argv[3]), nullptr};
    need(devices[0] && ggml_backend_dev_host_buffer_type(devices[0]), "requested Vulkan device and pinned storage unavailable");
    const bool watch = flags.count("--resources");
    need(!flags.count("--allocate-only") || (watch && flags.count("--warmup") && !flags.count("--verify")),
         "allocation-only mode requires warmup and resources, and holds no validation snapshots");
    if (watch) resource_sample("before-model", devices[0]);
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    const std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    need(bool(model), "could not load worker model");
    if (watch) resource_sample("after-model", devices[0]);
    auto cp = llama_context_default_params();
    cp.n_ctx = uint32_t(plan.context);
    cp.n_batch = cp.n_ubatch = chunk;
    cp.n_seq_max = 1;
    const std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    need(bool(ctx), "could not create single native execution context");
    if (watch) resource_sample("after-context", devices[0]);
    Worker worker(std::move(plan), ctx.get(), devices[0], std::stoull(argv[6]), argv[5], argv[7], std::chrono::milliseconds(ssd_hold_ms));
    if (short_limit) worker.configure_service(llama_model_get_vocab(model.get()), short_limit, long_limit);
    if (flags.count("--verify")) worker.qualification(size_t(llama_vocab_n_tokens(llama_model_get_vocab(model.get()))));
    if (flags.count("--warmup")) worker.warmup(flags.count("--warmup-grid"));
    if (watch) resource_sample("after-lane-warmup", devices[0]);
    if (flags.count("--allocate-only")) {
        worker.report_allocation();
        Sleep(4000); // Allow multiple WDDM counter snapshots of the same allocation.
        resource_sample("allocation-idle", devices[0]);
        return 0;
    }
    worker.run(flags.count("--quiet-events"), flags.count("--chunk-events"));
    if (watch) resource_sample("after-drain", devices[0]);
    return 0;
} catch (const std::exception &error) {
    std::cerr << "r10 catalogue worker at " << phase << ": " << error.what() << '\n';
    return 1;
}
