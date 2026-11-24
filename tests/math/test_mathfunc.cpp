// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "golden.hpp"
#include "golden_mathfunc.hpp"
#include <MatSAR/math/MathematicalFunction.hpp>
#include <MatSAR/math/Probability.hpp>   // mean / var (1-based dim)

using namespace matsar;
using namespace matsar_test;

// ---- Layer 1: golden replay (tag-grouped) ----
#define REPLAY_REDUCE(FN, ARR) \
    for (const auto &c: ARR) { \
        auto m = build<float64>(c.in, c.in_shape); \
        auto r = (c.axis == AXIS_NONE) ? Math::FN(m, none) \
                                       : Math::FN(m, (msize_type) c.axis); \
        check_matrix(c.name, r, c.out, c.out_shape, c.rtol, c.atol); \
    }

TEST_CASE("golden · sum"){REPLAY_REDUCE(sum, sum_golden)} TEST_CASE("golden · prod"){REPLAY_REDUCE(prod, prod_golden)} TEST_CASE("golden · cumsum"){REPLAY_REDUCE(cumsum, cumsum_golden)} TEST_CASE("golden · cumprod"){REPLAY_REDUCE(cumprod, cumprod_golden)}

// mean/var use MatSAR's 1-based `dim`, but the harness stores the 0-based axis (matching
// sum). Map it back at the call: axis case -> dim = axis + 1; global (AXIS_NONE) -> dim = 0
// (any dim <= 0 is global). var takes w before dim; use default population variance w=false.
#define REPLAY_REDUCE_1BASED(FN, ARR) \
    for (const auto &c: ARR) { \
        auto m = build<float64>(c.in, c.in_shape); \
        auto r = (c.axis == AXIS_NONE) ? Math::FN(m, (msize_type) 0) \
                                       : Math::FN(m, (msize_type) (c.axis + 1)); \
        check_matrix(c.name, r, c.out, c.out_shape, c.rtol, c.atol); \
    }

TEST_CASE("golden · mean") { REPLAY_REDUCE_1BASED(mean, mean_golden) }
TEST_CASE("golden · var") {
    for (const auto &c: var_golden) {
        auto m = build<float64>(c.in, c.in_shape);
        auto r = (c.axis == AXIS_NONE) ? Math::var(m, false, (msize_type) 0)
                                       : Math::var(m, false, (msize_type) (c.axis + 1));
        check_matrix(c.name, r, c.out, c.out_shape, c.rtol, c.atol);
    }
}

TEST_CASE("golden · sum (int64, exact)") {
    for (const auto &c: sum_golden_int) {
        auto m = build<int64>(c.in, c.in_shape);
        auto r = (c.axis == AXIS_NONE) ? Math::sum(m, none) : Math::sum(m, (msize_type) c.axis);
        check_matrix(c.name, r, c.out, c.out_shape, c.rtol, c.atol);
    }
}
TEST_CASE("golden · prod (int64, exact)") {
    for (const auto &c: prod_golden_int) {
        auto m = build<int64>(c.in, c.in_shape);
        auto r = (c.axis == AXIS_NONE) ? Math::prod(m, none) : Math::prod(m, (msize_type) c.axis);
        check_matrix(c.name, r, c.out, c.out_shape, c.rtol, c.atol);
    }
}

// ---- Layer 2: properties over seeded-random inputs (no oracle) ----
TEST_CASE("property · reductions vs brute force + identities") {
    Rng rng(12345);
    for (int t = 0; t < 200; ++t) {
        INFO("iter " << t);
        auto shape = random_shape(rng, 2, 1, 6);           // 2-D, dims 1..6
        auto m = random_matrix<float64>(rng, shape, -5.0, 5.0);

        // 1) global sum/prod match a brute-force reference
        CHECK(approx_eq(brute_sum(m), Math::sum(m, none)(0), 1e-10, 1e-12));
        CHECK(approx_eq(brute_prod(m), Math::prod(m, none)(0), 1e-9, 1e-12));

        // 2) reducing an axis then summing the result == global sum
        CHECK(approx_eq(reduce_all(Math::sum(m, (msize_type) 0)), Math::sum(m, none)(0), 1e-10, 1e-12));
        CHECK(approx_eq(reduce_all(Math::sum(m, (msize_type) 1)), Math::sum(m, none)(0), 1e-10, 1e-12));

        // 3) cumsum's last slice along an axis == sum along that axis
        auto cs = Math::cumsum(m, (msize_type) 1);
        auto sm = Math::sum(m, (msize_type) 1);              // shape {rows,1}
        const std::size_t rows = shape[0], cols = shape[1];
        for (std::size_t i = 0; i < rows; ++i) {
            CAPTURE(i);
            CHECK(approx_eq((double) cs(i, cols - 1), (double) sm(i, 0), 1e-10, 1e-12));
        }

        // 4) cumprod's last slice along an axis == prod along that axis
        auto cp = Math::cumprod(m, (msize_type) 1);
        auto pm = Math::prod(m, (msize_type) 1);
        for (std::size_t i = 0; i < rows; ++i) {
            CAPTURE(i);
            CHECK(approx_eq((double) cp(i, cols - 1), (double) pm(i, 0), 1e-10, 1e-12));
        }
    }
}
