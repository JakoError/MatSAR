// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Death tests for Probability: each case (selected by argv[1]) must be rejected with its
// own error message (MatSAR errors print and exit(); see tests/CMakeLists.txt, death_prob_*).
#include <MatSAR/math/Probability.hpp>
#include <cstdio>
#include <cstring>

using namespace matsar;

int main(int argc, char **argv) {
    const char *c = argc > 1 ? argv[1] : "";
    if (!std::strcmp(c, "randi_order"))
        Math::randi(5, 1, mshape_type{2});
    else if (!std::strcmp(c, "randperm_k"))
        Math::randperm<int64>(3, 4);
    else if (!std::strcmp(c, "randperm_negative"))
        Math::randperm<int64>(-1);
    else if (!std::strcmp(c, "normrnd_shape"))
        Math::normrnd(Matrix<float64>{0.0, 1.0}, Matrix<float64>{1.0, 1.0}, mshape_type{3});
    else if (!std::strcmp(c, "unifpdf_shape"))
        Math::unifpdf(Matrix<float64>{0.5}, Matrix<float64>{0.0, 1.0}, Matrix<float64>{1.0, 2.0, 3.0});
    else if (!std::strcmp(c, "pdf_shape"))
        Math::poisspdf(Matrix<float64>{1.0, 2.0}, Matrix<float64>{1.0, 2.0, 3.0});
    std::printf("no error for case '%s'\n", c);
    return 0;
}
