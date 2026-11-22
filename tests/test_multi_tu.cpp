// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Header-only library: MatSAR.hpp must be includable from several translation units of
// one program. This test links with tests/multi_tu_other.cpp, which includes it too; a
// non-inline function or variable defined in a header fails here with "multiple
// definition" at link time (the error helpers, fRange, matsar::none and the arange
// specializations all did).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <MatSAR/MatSAR.hpp>

using namespace matsar;

auto other_tu_arange_sum() -> int64;  // defined in multi_tu_other.cpp

TEST_CASE("MatSAR.hpp links from two translation units") {
    auto a = arange<int64>(0, 5, 1);  // 0 1 2 3 4, built in this translation unit
    int64 here = 0;
    for (const auto &v: a)
        here += v;
    CHECK(here == 10);
    CHECK(other_tu_arange_sum() == 10);  // the same computation in the other one
}
