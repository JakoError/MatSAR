// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Death tests for LinearAlgebra: each case (selected by argv[1]) must be rejected with its own
// error message (MatSAR errors print and exit(); see tests/CMakeLists.txt, death_linalg_*).
// The LAPACK-backed cases exist only with MATSAR_WITH_BLAS.
#include <MatSAR/math/LinearAlgebra.hpp>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace matsar;

int main(int argc, char **argv) {
    const char *c = argc > 1 ? argv[1] : "";
    if (!std::strcmp(c, "trace_nonsquare"))
        Math::trace(Matrix<float64>({{1., 2., 3.}, {4., 5., 6.}}));
    else if (!std::strcmp(c, "kron_3d"))
        Math::kron(Matrix<float64>(mshape_type{2, 2, 2}, 1.), Matrix<float64>({{1., 2.}, {3., 4.}}));
    else if (!std::strcmp(c, "ppval_pieces"))
        Math::ppval(std::vector<double>{0, 1, 2}, Matrix<double>({{1., 0.}, {2., 1.}, {3., 2.}}),
                    std::vector<double>{0.5});
    else if (!std::strcmp(c, "ppval_breaks"))
        Math::ppval(std::vector<double>{0}, Matrix<double>({{1., 0.}}), std::vector<double>{0.5});
#ifdef MATSAR_WITH_BLAS
    else if (!std::strcmp(c, "eig_complex_real"))
        Math::eig(Matrix<float64>({{1., 2.}, {-2., 1.}}));   // eigenvalues 1 +- 2i, real result type
    else if (!std::strcmp(c, "condeig_complex_real"))
        Math::condeig(Matrix<float64>({{1., 2.}, {-2., 1.}}));
    else if (!std::strcmp(c, "schur_flag"))
        Math::schur(Matrix<float64>({{1., 2.}, {3., 4.}}), "upper");
    else if (!std::strcmp(c, "schur_complex_real"))
        Math::schur(Matrix<float64>({{1., 2.}, {-2., 1.}}), "complex");
    else if (!std::strcmp(c, "det_nonsquare"))
        Math::det(Matrix<float64>({{1., 2., 3.}, {4., 5., 6.}}));
    else if (!std::strcmp(c, "inv_nonsquare"))
        Math::inv(Matrix<float64>({{1., 2., 3.}, {4., 5., 6.}}));
    else if (!std::strcmp(c, "pinv_3d"))
        Math::pinv(Matrix<float64>(mshape_type{2, 2, 2}, 1.));
    else if (!std::strcmp(c, "lu_3d"))
        Math::lu(Matrix<float64>(mshape_type{2, 2, 2}, 1.));
    else if (!std::strcmp(c, "svd_1d"))
        Math::svd(Matrix<float64>({1., 2., 3.}));
    else if (!std::strcmp(c, "polyvalm_nonsquare")) {
        const Matrix<double> X({{1., 2., 3.}, {4., 5., 6.}});
        Math::polyvalm(std::vector<double>{1, 2}, X);
    }
#endif
    std::printf("no error for case '%s'\n", c);
    return 0;
}
