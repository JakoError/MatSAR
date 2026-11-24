// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Second translation unit of test_math_multi_tu (see multi_tu_a.cpp).
#include "multi_tu_headers.hpp"

auto other_tu_mean() -> double {
    matsar::Matrix<double> m = {1.0, 2.0, 3.0, 6.0};
    return matsar::Math::mean(m, 0)(0);
}
