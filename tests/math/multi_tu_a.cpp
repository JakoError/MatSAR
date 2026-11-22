// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Every experimental math header must be includable from several translation units of
// one program: this file and multi_tu_b.cpp include them all and are linked together,
// so a non-inline function or variable defined in a header fails with "multiple
// definition" at link time (LogicFunctions, LinearAlgebra and Probability all had them).
#include "multi_tu_headers.hpp"
#include <cstdio>

auto other_tu_mean() -> double;  // defined in multi_tu_b.cpp

int main() {
    matsar::Matrix<double> m = {1.0, 2.0, 3.0, 6.0};
    double here = matsar::Math::mean(m, 0)(0);
    double there = other_tu_mean();
    std::printf("mean here %g, in the other translation unit %g\n", here, there);
    return here == 3.0 && there == 3.0 ? 0 : 1;
}
