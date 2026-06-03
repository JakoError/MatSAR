// broadcasting.cpp -- element-wise arithmetic, broadcasting, and comparisons.
//
// IMPORTANT: between two matrices, `*` is ELEMENT-WISE (NumPy semantics), not a
// matrix product. The matrix product is mmul(a, b) / a % b (see matmul.cpp).

#include <iostream>

#include <MatSAR/MatSAR.hpp>

using namespace matsar;

int main() {
    auto a = Matrix<float64>::fromRows({{1, 2, 3},
                                        {4, 5, 6}});   // 2x3

    // Scalar operands apply to every element (either side works).
    std::cout << (a + 10).toRowColString("a + 10");
    std::cout << (2 * a).toRowColString("2 * a");

    // Element-wise between equal shapes.
    auto b = Matrix<float64>::fromRows({{6, 5, 4},
                                        {3, 2, 1}});
    std::cout << (a * b).toRowColString("a * b   (element-wise!)");
    std::cout << (a / b).toRowColString("a / b");

    // Broadcasting: a one-row matrix ({1,3}) is reused across every row of a.
    auto row = Matrix<float64>::fromRows({{100, 200, 300}});   // shape {1,3}
    std::cout << (a + row).toRowColString("a + row   (broadcast over rows)");

    // Comparisons broadcast too and yield a boolean matrix.
    std::cout << (a > 3).toRowColString("a > 3");

    return 0;
}
