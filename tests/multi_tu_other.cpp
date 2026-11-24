// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Second translation unit of test_multi_tu (see test_multi_tu.cpp).
#include <MatSAR/MatSAR.hpp>

using namespace matsar;

auto other_tu_arange_sum() -> int64 {
    auto a = arange<int64>(0, 5, 1);
    int64 sum = 0;
    for (const auto &v: a)
        sum += v;
    return sum;
}
