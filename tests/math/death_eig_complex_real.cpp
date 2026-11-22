// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Death test: eig() with a real result type on a matrix whose eigenvalues are complex
// must refuse (ValueError) instead of silently returning only the real parts.
// MatSAR errors print a message and exit() rather than throw, so CTest matches it.
#include <MatSAR/math/LinearAlgebra.hpp>

using namespace matsar;

int main() {
    Matrix<float64> A = {{1, 2}, {-2, 1}};   // eigenvalues 1 +/- 2i
    auto VDW = Math::eig(A);                  // default result type: float64
    (void) VDW;
    return 0;                                 // not reached
}
