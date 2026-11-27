#include "r10_catalog_ledger.h"

#include <iostream>

using r10cat::PayloadLedger;
constexpr size_t P = R10StagePool::page;

static void expect(bool ok) { if (!ok) throw std::runtime_error("lease audit assertion failed"); }
template<typename T> static void reject(T call) {
    bool rejected = false;
    try { call(); } catch (const std::logic_error &) { rejected = true; }
    expect(rejected);
}

int main() try {
    // Head is too large once another request owns some of the pool. Its private
    // upload becomes shared after reservation while that upload is in flight.
    {
        PayloadLedger p(3 * P, {P, P, P, P, P});
        p.bind_head(0, {0, 1, 2}, {});
        expect(p.dispatch(0, 0, true).direct);
        p.submitted(1);
        p.reserve(1, {0, 3}, {});
        p.reserve(2, {0, 4}, {});
        expect(p.pool().charged() == 3 * P && p.pool().get(0).refs == 2 && !p.aliases().count(0));
        reject([&] { p.source(0); });
        reject([&] { p.dispatch(1, 3, false); });
        reject([&] { p.landed(99); });
        p.landed(1);
        expect(p.source(0) == PayloadLedger::Source::active && p.aliases().count(0));
        for (size_t id = 1; id <= 2; ++id) {
            expect(p.dispatch(0, id, true).direct);
            p.submitted(id + 1);
            p.landed(id + 1);
        }
        p.begin_compute(0, false, true);
        expect(!p.dispatch(1, 3, false).direct);
        p.submitted(4);
        p.computed(0);
        p.bind_head(1, {0, 3}, {0});
        expect(p.source(0) == PayloadLedger::Source::active);
        reject([&] { p.source(3); });
        p.landed(4);
        p.scattered(3);
        p.begin_compute(1, true, true);
        p.computed(1);
        expect(p.pool().charged() == 2 * P && p.before_switch({}) == std::vector<size_t>{0});
        reject([&] { p.bind_head(2, {0, 4}, {}); });
        p.materialized(0);
        p.bind_head(2, {0, 4}, {});
        expect(p.source(0) == PayloadLedger::Source::staged);
        p.scattered(0);
        expect(!p.dispatch(2, 4, true).direct);
        p.submitted(5);
        p.landed(5);
        p.scattered(4);
        p.begin_compute(2, true, true);
        p.computed(2);
        expect(p.drained());
    }
    // A sibling can reserve a private recurrent upload while in flight. The
    // stationary snapshot permits the alias, but suffix writes require a copy.
    {
        PayloadLedger p(2 * P, {P, P});
        p.bind_head(0, {0, 1}, {});
        expect(p.dispatch(0, 0, true).direct);
        p.submitted(1);
        p.landed(1);
        expect(p.dispatch(0, 1, true).direct);
        p.submitted(2);
        p.reserve(1, {0, 1}, {});
        p.landed(2);
        expect(p.recurrent_must_preserve(false, true) && !p.recurrent_must_preserve(false, false));
        reject([&] { p.begin_compute(0, false, true); });
        p.materialized(1);
        p.begin_compute(0, false, true);
        p.computed(0);
        p.bind_head(1, {0, 1}, {0});
        expect(p.source(0) == PayloadLedger::Source::active && p.source(1) == PayloadLedger::Source::staged);
        p.scattered(1);
        p.begin_compute(1, true, true);
        p.computed(1);
        expect(p.drained());
    }
    // A fully recomputed request must not overwrite a borrowed KV block still
    // charged to a nonadjacent consumer.
    {
        PayloadLedger p(2 * P, {P, P, P});
        p.bind_head(0, {0, 1}, {});
        for (size_t id = 0; id < 2; ++id) {
            expect(p.dispatch(0, id, true).direct);
            p.submitted(id + 1);
            p.landed(id + 1);
        }
        p.begin_compute(0, false, true);
        p.reserve(2, {0, 2}, {0});
        p.computed(0);
        expect(p.before_switch({}) == std::vector<size_t>{0});
        p.materialized(0);
        p.bind_head(1, {}, {});
        p.begin_compute(1, false, true);
        expect(!p.dispatch(2, 2, false).direct);
        p.submitted(3);
        p.landed(3);
        p.computed(1);
        p.bind_head(2, {0, 2}, {});
        p.scattered(0);
        p.scattered(2);
        p.begin_compute(2, true, true);
        p.computed(2);
        expect(p.drained());
    }
    std::cout << "Checked mixed head/staging, in-flight handoff, shared recurrent copy, and nonadjacent KV loans.\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << "r10 ledger probe: " << error.what() << '\n';
    return 1;
}
