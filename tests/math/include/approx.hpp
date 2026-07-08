// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
#ifndef MATSAR_TEST_APPROX_HPP
#define MATSAR_TEST_APPROX_HPP
#include <cmath>

namespace matsar_test {
// NumPy allclose-style comparison with NaN/Inf awareness.
    inline bool approx_eq(double got, double want, double rtol, double atol) {
        if (std::isnan(want)) return std::isnan(got);   // domain-error cases: NaN matches NaN
        if (std::isinf(want)) return got == want;        // sign-correct Inf
        return std::abs(got - want) <= atol + rtol * std::abs(want);
    }
}  // namespace matsar_test
#endif
