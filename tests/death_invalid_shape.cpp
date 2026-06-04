// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Death test: an incompatible-shape op prints a "*Exception" message and exit()s.
// MatSAR's current error handling EXITS rather than throwing a catchable C++
// exception (see ExceptionUtil.hpp; "better exception handling" is future work),
// so this is verified as a death test via CTest PASS_REGULAR_EXPRESSION, not
// doctest CHECK_THROWS.
#include <MatSAR/Matrix.hpp>

using namespace matsar;

int main() {
    Matrix<int64> a = {{1, 2, 3}, {4, 5, 6}};  // shape {3,2}
    Matrix<int64> b = {{1, 2}, {3, 4}};         // shape {2,2}
    auto c = a + b;                              // non-broadcastable -> message + exit()
    (void) c;
    return 0;                                    // not reached
}
