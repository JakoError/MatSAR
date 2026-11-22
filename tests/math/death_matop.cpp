// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Death tests for MatrixOperation: each case (selected by argv[1]) must be rejected
// with its own ValueError. MatSAR errors print a message and exit() rather than throw,
// so CTest matches the printed message (see tests/CMakeLists.txt, death_matop_*).
#include <MatSAR/math/MatrixOperation.hpp>
#include <cstdio>
#include <cstring>

using namespace matsar;

int main(int argc, char **argv) {
    const char *c = argc > 1 ? argv[1] : "";
    Matrix<int64> m = {{1, 2, 3}, {4, 5, 6}};
    Matrix<float64> sq = {{1, 2}, {3, 4}};
    if (!std::strcmp(c, "ipermute_range"))
        Math::ipermute(m, {3, 1});                        // 3 > number of dimensions
    else if (!std::strcmp(c, "ipermute_repeated"))
        Math::ipermute(m, {1, 1});
    else if (!std::strcmp(c, "factorial_nonint"))
        Math::factorial(Matrix<float64>{2.5});
    else if (!std::strcmp(c, "factorial_negative"))
        Math::factorial(Matrix<int64>{-3});
    else if (!std::strcmp(c, "numden_zero"))
        Math::numden(Matrix<int64>{0}, Matrix<int64>{0});
    else if (!std::strcmp(c, "expm_nonsquare"))
        Math::expm(Matrix<float64>{{1, 2, 3}, {4, 5, 6}});
    else if (!std::strcmp(c, "logm_negative"))
        Math::logm(Matrix<float64>{{-1.0, 0.0}, {0.0, 2.0}});     // real result requested, log(-1) is complex
    else if (!std::strcmp(c, "logm_singular"))
        Math::logm(Matrix<float64>{{1, 2}, {2, 4}});
    else if (!std::strcmp(c, "mpower_singular"))
        Math::mpower(Matrix<float64>{{1, 2}, {2, 4}}, -1);
    else if (!std::strcmp(c, "mpower_two_matrices"))
        Math::mpower(sq, sq);
    else if (!std::strcmp(c, "mpower_negative_base"))
        Math::mpower(-2.0, sq);                           // real result requested, log(-2) is complex
    std::printf("no error for case '%s'\n", c);
    return 0;
}
