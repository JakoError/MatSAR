// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <MatSAR/MatSAR.hpp>   // default (experimental OFF) public header must compile

using namespace matsar;

TEST_CASE("public header exposes the core") {
    auto m = ones({2, 2});
    CHECK(m(0, 0) == doctest::Approx(1.0));
}
