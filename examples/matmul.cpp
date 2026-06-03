// matmul.cpp -- the BLAS-backed matrix product, mmul(a, b) and its alias a % b.
//
// This example needs the OpenBLAS-backed build (the default, MATSAR_WITH_BLAS
// ON). CMake only adds it to the build when BLAS is enabled; the #ifdef below
// keeps the file self-contained if you compile it by hand without BLAS.

#include <iostream>

#include <MatSAR/MatSAR.hpp>

using namespace matsar;

int main() {
#ifdef MATSAR_WITH_BLAS
    // (2x3) * (3x2) -> (2x2). Remember: `*` would be element-wise and would
    // reject these shapes; the matrix product is mmul / operator%.
    auto a = Matrix<float64>::fromRows({{1, 2, 3},
                                        {4, 5, 6}});      // 2x3
    auto b = Matrix<float64>::fromRows({{7, 8},
                                        {9, 10},
                                        {11, 12}});       // 3x2

    std::cout << mmul(a, b).toRowColString("mmul(a, b)");
    std::cout << (a % b).toRowColString("a % b   (same thing)");
    return 0;
#else
    std::cout << "Built without OpenBLAS (MATSAR_WITH_BLAS=OFF): "
                 "mmul/operator% are unavailable.\n";
    return 0;
#endif
}
