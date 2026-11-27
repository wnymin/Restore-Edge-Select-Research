#pragma once

#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace r10cat {
using Bytes = std::vector<uint8_t>;
using Tokens = std::vector<llama_token>;
using Shape = std::array<llama_r10_restore_span, 512>;
namespace fs = std::filesystem;
inline void need(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }

struct ObjectSpec { char kind = 0; size_t bytes = 0; Tokens prefix; };
struct Candidate { size_t depth = 0; std::vector<size_t> deps; };
struct Row { Tokens tokens; std::map<size_t, Candidate> options; };
struct Plan {
    size_t block = 0, context = 0;
    std::vector<ObjectSpec> objects;
    std::vector<Row> requests;
    std::map<size_t, size_t> checkpoint_owner;
    std::map<size_t, size_t> checkpoint_depth;
};

inline Plan read_plan(const fs::path &path) {
    std::ifstream f(path);
    Plan p;
    std::string magic;
    size_t objects = 0, requests = 0;
    need(bool(f >> magic >> p.block >> p.context >> objects >> requests) && magic == "R10CAT1" &&
         p.block == 512 && p.context == 3072 && objects < 10000 && requests > 0 && requests <= 4096,
         "invalid R10 plan header");
    p.objects.resize(objects);
    for (auto &o : p.objects) need(bool(f >> o.kind >> o.bytes) &&
        ((o.kind == 'K' && o.bytes == 16777216) || (o.kind == 'R' && o.bytes == 52690944)), "invalid R10 object shape");
    p.requests.resize(requests);
    for (size_t i = 0; i < requests; ++i) {
        size_t n = 0, options = 0;
        auto &r = p.requests[i];
        need(bool(f >> n >> options) && n && n <= p.context - 16 && options && options <= n / p.block + 1,
             "invalid R10 request size");
        r.tokens.resize(n);
        for (auto &token : r.tokens) need(bool(f >> token) && token >= 0 && token < 250000, "invalid R10 token id");
        size_t previous = 0;
        for (size_t j = 0; j < options; ++j) {
            size_t d = 0, count = 0;
            need(bool(f >> d >> count) && d <= n && d % p.block == 0 && (j ? d > previous : d == 0) &&
                 count == (d ? d / p.block + 1 : 0), "invalid R10 candidate layout");
            Candidate c{d, std::vector<size_t>(count)};
            for (size_t k = 0; k < count; ++k) {
                auto &id = c.deps[k];
                need(bool(f >> id) && id < objects, "invalid R10 dependency id");
                auto &o = p.objects[id];
                const bool kv = k + 1 < count;
                const size_t end = kv ? (k + 1) * p.block : d;
                need(o.kind == (kv ? 'K' : 'R'), "wrong R10 dependency kind");
                if (o.prefix.empty()) o.prefix.assign(r.tokens.begin(), r.tokens.begin() + end);
                need(o.prefix.size() == end && std::equal(o.prefix.begin(), o.prefix.end(), r.tokens.begin()),
                     "R10 dependency changed prefix identity");
            }
            if (d) {
                p.checkpoint_owner.emplace(c.deps.back(), i);
                p.checkpoint_depth.emplace(c.deps.back(), d);
            }
            r.options.emplace(d, std::move(c));
            previous = d;
        }
    }
    std::map<std::pair<char, Tokens>, size_t> canonical;
    for (size_t id = 0; id < p.objects.size(); ++id) {
        const auto &o = p.objects[id];
        need(!o.prefix.empty() && canonical.emplace(std::make_pair(o.kind, o.prefix), id).second,
             "catalogue has an unreferenced object or split canonical identity");
    }
    need(!(f >> magic) && f.eof(), "trailing content in R10 plan");
    return p;
}

struct Part { size_t object, offset, bytes; };
struct Region { size_t source, bytes; std::string tensor; std::vector<Part> parts; };
struct Meta { size_t depth = 0; Bytes skeleton; std::vector<Region> regions; };

inline fs::path path_for(const fs::path &root, size_t id, bool meta) {
    return root / (std::to_string(id) + (meta ? ".meta" : ".bin"));
}
template<typename T> inline void write_scalar(std::ostream &f, T value) {
    f.write(reinterpret_cast<const char *>(&value), sizeof(T));
    need(bool(f), "could not write R10 metadata");
}
template<typename T> inline T read_scalar(std::istream &f) {
    T value{};
    f.read(reinterpret_cast<char *>(&value), sizeof(T));
    need(bool(f), "truncated R10 metadata");
    return value;
}

inline std::vector<const Region *> ordered(const Meta &m) {
    std::vector<const Region *> order;
    for (auto &r : m.regions) order.push_back(&r);
    std::sort(order.begin(), order.end(), [](auto a, auto b) { return a->source < b->source; });
    size_t offset = 0;
    for (auto r : order) {
        need(r->source >= offset && r->source <= m.skeleton.size() && r->bytes && r->bytes <= m.skeleton.size() - r->source,
             "overlapping or invalid R10 metadata hole");
        offset = r->source + r->bytes;
    }
    return order;
}

inline void store_meta(const fs::path &path, const Meta &m) {
    need(!fs::exists(path), "R10 requires a new metadata path");
    std::ofstream f(path, std::ios::binary);
    f.write("R10META1", 8);
    write_scalar<uint32_t>(f, uint32_t(m.depth));
    write_scalar<uint64_t>(f, m.skeleton.size());
    write_scalar<uint32_t>(f, uint32_t(m.regions.size()));
    for (auto &r : m.regions) {
        write_scalar<uint64_t>(f, r.source);
        write_scalar<uint64_t>(f, r.bytes);
        write_scalar<uint32_t>(f, uint32_t(r.tensor.size()));
        f.write(r.tensor.data(), r.tensor.size());
        write_scalar<uint32_t>(f, uint32_t(r.parts.size()));
        for (auto &s : r.parts) {
            write_scalar<uint32_t>(f, uint32_t(s.object));
            write_scalar<uint64_t>(f, s.offset);
            write_scalar<uint64_t>(f, s.bytes);
        }
    }
    size_t offset = 0;
    for (auto r : ordered(m)) {
        f.write(reinterpret_cast<const char *>(m.skeleton.data() + offset), r->source - offset);
        offset = r->source + r->bytes;
    }
    f.write(reinterpret_cast<const char *>(m.skeleton.data() + offset), m.skeleton.size() - offset);
    f.close();
    need(bool(f), "could not store R10 compact restore skeleton");
}

inline Meta load_meta(const fs::path &path, const Plan &p, const Candidate &c) {
    need(c.depth > 0 && c.deps.size() >= 2, "zero-depth candidates have no restore metadata");
    std::ifstream f(path, std::ios::binary);
    std::array<char, 8> magic{};
    need(bool(f.read(magic.data(), magic.size())) && std::string(magic.data(), magic.size()) == "R10META1", "unknown R10 metadata");
    Meta m;
    m.depth = read_scalar<uint32_t>(f);
    const auto image = read_scalar<uint64_t>(f);
    const auto n = read_scalar<uint32_t>(f);
    size_t payload = 0;
    for (auto id : c.deps) payload += p.objects.at(id).bytes;
    need(m.depth == c.depth && image > payload && image < payload + (2 << 20) && n && n <= Shape{}.size(), "bad R10 metadata size");
    m.skeleton.resize(size_t(image));
    std::map<size_t, size_t> used;
    for (uint32_t i = 0; i < n; ++i) {
        Region r{};
        r.source = read_scalar<uint64_t>(f);
        r.bytes = read_scalar<uint64_t>(f);
        const auto length = read_scalar<uint32_t>(f);
        need(length && length < GGML_MAX_NAME, "bad R10 tensor name size");
        r.tensor.resize(length);
        need(bool(f.read(r.tensor.data(), length)), "bad R10 tensor name");
        const auto count = read_scalar<uint32_t>(f);
        need(count && count <= c.deps.size(), "bad R10 span part count");
        size_t covered = 0;
        for (uint32_t k = 0; k < count; ++k) {
            Part s{read_scalar<uint32_t>(f), read_scalar<uint64_t>(f), read_scalar<uint64_t>(f)};
            need(std::find(c.deps.begin(), c.deps.end(), s.object) != c.deps.end() && s.bytes && s.bytes % 4 == 0 &&
                 s.offset == used[s.object] && s.offset <= p.objects[s.object].bytes &&
                 s.bytes <= p.objects[s.object].bytes - s.offset, "bad R10 span object coverage");
            used[s.object] += s.bytes;
            covered += s.bytes;
            r.parts.push_back(s);
        }
        need(covered == r.bytes, "missing R10 span bytes");
        if (r.parts.front().object == c.deps.back()) {
            need(r.parts.size() == 1, "mixed attention and recurrent tensor");
        } else {
            need(r.parts.size() + 1 == c.deps.size(), "attention tensor does not cover every block");
            for (size_t j = 0; j < r.parts.size(); ++j) {
                need(r.parts[j].object == c.deps[j] && r.parts[j].bytes == r.parts.front().bytes,
                     "attention tensor has misordered or unequal KV blocks");
            }
        }
        m.regions.push_back(std::move(r));
    }
    for (auto id : c.deps) need(used[id] == p.objects[id].bytes, "incomplete R10 object coverage");
    size_t offset = 0;
    for (auto r : ordered(m)) {
        need(bool(f.read(reinterpret_cast<char *>(m.skeleton.data() + offset), r->source - offset)), "truncated R10 metadata gap");
        offset = r->source + r->bytes;
    }
    need(bool(f.read(reinterpret_cast<char *>(m.skeleton.data() + offset), m.skeleton.size() - offset)) &&
         f.peek() == std::char_traits<char>::eof(), "bad R10 compact skeleton tail");
    return m;
}
}
