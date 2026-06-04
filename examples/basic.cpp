// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// basic.cpp -- constructing, indexing, and printing matrices.
//
// Build this example by configuring with -DMATSAR_BUILD_EXAMPLES=ON (the
// default when MatSAR is the top-level project), then run ./examples/basic.

#include <iostream>

#include <MatSAR/MatSAR.hpp>

using namespace matsar;

int main() {
    // A nested initializer list is read ROW-wise by default (NumPy/MATLAB
    // intuition): {{1,2,3},{4,5,6}} has rows [1,2,3] and [4,5,6], so this is a
    // 2x3 matrix with a(0,1) == 2 and a(1,0) == 4.
    Matrix<int64> a = {{1, 2, 3},
                       {4, 5, 6}};

    std::cout << a.toString("a");                 // data-structure (braced) view
    std::cout << "a is " << a.getShape()[0] << "x" << a.getShape()[1]
              << ", a(0,1) = " << a(0, 1)
              << ", a(1,0) = " << a(1, 0) << "\n\n";

    // Negative indices count from the end, like Python: (-1,-1) is the last
    // element.
    std::cout << "a(-1,-1) = " << a(-1, -1) << "\n\n";

    // Build the same data explicitly. fromRows is the default; fromColumns
    // treats each inner list as a column (the legacy layout), so it is the
    // transpose of fromRows for the same braces.
    auto byRows = Matrix<int64>::fromRows({{1, 2, 3}, {4, 5, 6}});       // 2x3
    auto byCols = Matrix<int64>::fromColumns({{1, 2, 3}, {4, 5, 6}});    // 3x2
    std::cout << byRows.toRowColString("byRows");
    std::cout << byCols.toRowColString("byCols");

    // A flat initializer list is a 1-D vector (shape {N}).
    Matrix<int64> v = {10, 20, 30};
    std::cout << "v has size " << v.getSize() << ", v(2) = " << v(2) << "\n\n";

    // Creation helpers (see <MatSAR/MatrixCreation.hpp>).
    std::cout << eye<float64>(3).toRowColString("eye(3)");
    std::cout << ones<float64>({2, 2}).toRowColString("ones({2,2})");

    return 0;
}
