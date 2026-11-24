// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Death test: permute() with a repeated dimension must be rejected with its
// "repeated" ValueError. MatSAR errors print a message and exit() rather than
// throw (see tests/death_invalid_shape.cpp), so CTest matches the message.
// Before the fix the duplicate slipped past the guard and the copy loop died later
// with an unrelated Out_Of_Range read, which this test must not accept.
#include <MatSAR/math/MatrixOperation.hpp>

using namespace matsar;

int main() {
    Matrix<int64> m = {{1, 2, 3}, {4, 5, 6}};
    auto r = Math::permute(m, {1, 1});   // dimension 1 used twice
    (void) r;
    return 0;                            // not reached
}
