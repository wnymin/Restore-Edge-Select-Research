#include "r10_catalog.h"

#include <iostream>
#include <numeric>

// This host-only audit does not link a llama or GPU DLL.
int main(int argc, const char **argv) try {
    using namespace r10cat;
    need(argc == 2 || argc == 5, "usage: r10_catalog_inspect PLAN [REQUEST DEPTH META]");
    const auto p = read_plan(argv[1]);
    if (argc == 5) {
        const auto &c = p.requests.at(std::stoull(argv[2])).options.at(std::stoull(argv[3]));
        need(c.depth != 0, "zero-depth candidates have no descriptor");
        const auto m = load_meta(argv[4], p, c);
        size_t payload = 0;
        for (const auto &r : m.regions) payload += r.bytes;
        const size_t gaps = m.skeleton.size() - payload;
        std::cout << m.depth << ' ' << m.regions.size() << ' ' << payload << ' ' << gaps << '\n';
        return 0;
    }
    size_t tokens = 0, choices = 0;
    std::vector<size_t> refs(p.objects.size());
    std::map<size_t, size_t> depths;
    for (const auto &row : p.requests) {
        tokens += row.tokens.size();
        for (const auto &[depth, option] : row.options) {
            ++depths[depth];
            ++choices;
            for (auto id : option.deps) ++refs.at(id);
        }
    }
    std::cout << p.requests.size() << ' ' << p.objects.size() << ' ' << p.checkpoint_owner.size() << ' ' << tokens << ' '
              << choices << ' ' << std::count_if(refs.begin(), refs.end(), [](auto n) { return n > 1; }) << '\n';
    for (const auto &[depth, n] : depths) std::cout << depth << ' ' << n << '\n';
    return 0;
} catch (const std::exception &error) {
    std::cerr << "r10 catalogue rejected: " << error.what() << '\n';
    return 1;
}
