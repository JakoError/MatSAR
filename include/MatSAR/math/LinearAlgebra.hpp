// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2022/7/9.
//
///LA Linear Algebra
/// kron, geomean, harmmean, trace, the polynomial helpers, ppval, Expint and spdiags need nothing
/// beyond the core. svd, rank, lu, lup, det, inv, eig, condeig, pinv, schur, polyvalm and funm use
/// LAPACKE / CBLAS and exist only with MATSAR_WITH_BLAS.
#ifndef MATSAR_LINEARALGEBRA_HPP
#define MATSAR_LINEARALGEBRA_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

#ifndef MATSAR_MATRIXCREATION_HPP

#include "../MatrixCreation.hpp"

#endif //MATSAR_MATRIXCREATION_HPP

#ifndef MATSAR_UNIVERSALFUNCTIONS_HPP

#include "../UniversalFunctions.hpp"

#endif //MATSAR_UNIVERSALFUNCTIONS_HPP

#ifndef MATSAR_MATHEMATICALFUNCTION_HPP

#include "MathematicalFunction.hpp"

#endif //MATSAR_MATHEMATICALFUNCTION_HPP

#ifdef MATSAR_WITH_BLAS

// LAPACKE is used by this module's SVD / LU / inv / eig routines. Configure the
// complex types and integer width, then pull in the header. (Relocated here from
// Matrix.hpp so the core depends only on CBLAS; <lapacke.h> lives with the code
// that actually uses it.)
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>

// have to be int
#define lapack_int int

#ifndef _LAPACKE_H_

#include <lapacke.h>

#endif

#endif // MATSAR_WITH_BLAS

using matsar::Util::TypeUtil::eps;

using matsar::Util::VectorUtil::Range;

namespace matsar {
    namespace Math {
        template<typename T1, typename T2, typename TargetType = decltype(std::declval<T1>() * std::declval<T2>())>
        auto kron(Matrix<T1> m1, Matrix<T2> m2) -> Matrix<TargetType> {
            m1.expandShape(2);
            m2.expandShape(2);
            if (!m1.is2D() || !m2.is2D())
                throw_value_error(
                        "kron: kron requires that both matrices be two-dimensional or extensible to two-dimensional matrices");
            auto shape1 = m1.getShape();
            auto shape2 = m2.getShape();
            mshape_type r_shape = {shape1[0] * shape2[0], shape1[1] * shape2[1]};
            Matrix<TargetType> r(r_shape, TargetType(0), m1.getData());
            for (msize_type i = 0; i < shape1[0]; ++i) {
                for (msize_type j = 0; j < shape1[1]; ++j) {
                    for (msize_type k = 0; k < shape2[0]; ++k) {
                        for (msize_type l = 0; l < shape2[1]; ++l) {
                            r(i * shape2[0] + k, j * shape2[1] + l) = m1(i, j) * m2(k, l);
                        }
                    }
                }
            }
            return r;
        }
    }

#ifdef MATSAR_WITH_BLAS
    namespace Math {
        namespace Internal {
            inline auto LAPACK_SVD(double *a, double *s, double *u, double *vt,
                            char jobu, char jobvt,
                            lapack_int m, lapack_int n,
                            lapack_int lda,
                            lapack_int ldu, lapack_int ldvt,
                            double *work, lapack_int lwork) -> lapack_int {
                return LAPACKE_dgesvd_work(CblasColMajor, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
            }

            inline auto LAPACK_SVD(lapack_complex_float *a, float *s, lapack_complex_float *u, lapack_complex_float *vt,
                            char jobu, char jobvt,
                            lapack_int m, lapack_int n,
                            lapack_int lda,
                            lapack_int ldu,
                            lapack_int ldvt, lapack_complex_float *work,
                            lapack_int lwork, float *rwork) -> lapack_int {
                return LAPACKE_cgesvd_work(CblasColMajor, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                                           rwork);
            }

            inline auto LAPACK_SVD(lapack_complex_double *a, double *s, lapack_complex_double *u, lapack_complex_double *vt,
                            char jobu, char jobvt,
                            lapack_int m, lapack_int n,
                            lapack_int lda,
                            lapack_int ldu,
                            lapack_int ldvt, lapack_complex_double *work,
                            lapack_int lwork, double *rwork) -> lapack_int {
                return LAPACKE_zgesvd_work(CblasColMajor, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                                           rwork);
            }

            template<typename>
            auto LAPACK_SVD_WORK_SIZE(char jobu, char jobvt,
                                      lapack_int m, lapack_int n,
                                      lapack_int lda,
                                      lapack_int ldu, lapack_int ldvt) -> lapack_int = delete;

            template<>
            inline auto LAPACK_SVD_WORK_SIZE<float64>(char jobu, char jobvt,
                                               lapack_int m, lapack_int n,
                                               lapack_int lda,
                                               lapack_int ldu, lapack_int ldvt) -> lapack_int {
                unique_ptr<float64> size(new float64);
                LAPACK_SVD(nullptr, nullptr, nullptr, nullptr,
                           jobu, jobvt, m, n, lda, ldu, ldvt,
                           size.get(), -1);
                return static_cast<lapack_int>(std::ceil(*size));
            }

            template<>
            inline auto LAPACK_SVD_WORK_SIZE<complex64>(char jobu, char jobvt,
                                                 lapack_int m, lapack_int n,
                                                 lapack_int lda,
                                                 lapack_int ldu,
                                                 lapack_int ldvt) -> lapack_int {
                unique_ptr<complex64> size(new complex64);
                LAPACK_SVD(nullptr, nullptr, nullptr, nullptr,
                           jobu, jobvt, m, n, lda, ldu, ldvt,
                           size.get(), -1, nullptr);
                return static_cast<lapack_int>(std::ceil(std::real(*size)));
            }

            template<>
            inline auto LAPACK_SVD_WORK_SIZE<complex128>(char jobu, char jobvt,
                                                  lapack_int m, lapack_int n,
                                                  lapack_int lda,
                                                  lapack_int ldu,
                                                  lapack_int ldvt) -> lapack_int {
                unique_ptr<complex128> size(new complex128);
                LAPACK_SVD(nullptr, nullptr, nullptr, nullptr,
                           jobu, jobvt, m, n, lda, ldu, ldvt,
                           size.get(), -1, nullptr);
                return static_cast<lapack_int>(std::ceil(std::real(*size)));
            }

            template<typename dType>
            void svd_prepare_uv(const Matrix<dType> &A, const bool &compute_uv, const bool &full_matrices, int m, int n,
                                const int &k,
                                char &jobu, char &jobvt, int &ldu, int &ldvt, Matrix<dType> &U, Matrix<dType> &V) {
                // as U V have the same type in every condition, extract to this function
                if (compute_uv) {
                    auto U_shape = A.getShape();
                    auto V_shape = A.getShape();
                    if (full_matrices) {
                        U_shape[0] = m;
                        U_shape[1] = m;
                        V_shape[0] = n;
                        V_shape[1] = n;
                        jobu = 'A';
                        jobvt = 'A';
                        ldu = m;
                        ldvt = n;
                    } else {
                        U_shape[0] = m;
                        U_shape[1] = k;
                        V_shape[0] = k;
                        V_shape[1] = n;
                        jobu = 'S';
                        jobvt = 'S';
                        ldu = m;
                        ldvt = k;
                    }
                    U = zeros<dType>(U_shape);
                    V = zeros<dType>(V_shape);
                } else {
                    jobu = 'N';
                    jobvt = 'N';
                    ldu = 1;
                    ldvt = 1;
                }
            }

            inline auto do_svd(const Matrix<float64> &A, const bool &compute_uv, const bool &full_matrices) -> vector<Matrix<float64>> {
                using T = float64;

                auto m = static_cast<lapack_int>(A.getShape()[0]), n = static_cast<lapack_int>(A.getShape()[1]);

                auto &k = std::min(m, n);
                mshape_type S_shape = Util::VectorUtil::cut(A.getShape(), 1);
                S_shape[0] = k;
                auto S = matsar::zeros<T>(S_shape);

                char jobu, jobvt;
                lapack_int lda = m, ldu, ldvt;
                Matrix<T> U, V;
                svd_prepare_uv(A, compute_uv, full_matrices, m, n, k, jobu, jobvt, ldu, ldvt, U, V);

                auto work_size = LAPACK_SVD_WORK_SIZE<float64>(jobu, jobvt, m, n, lda, ldu, ldvt);
                unique_ptr<float64[]> work(new T[work_size]);

                auto unit_num = A.getUnitNum();
                auto A_unit_size = A.getUnitSize();
                auto S_unit_size = k;
                auto U_unit_size = U.getUnitSize();
                auto V_unit_size = V.getUnitSize();

                auto A_iter = A.begin();
                auto S_iter = S.begin();
                auto U_iter = U.begin();
                auto V_iter = V.begin();

                for (decltype(unit_num) i = 0; i < unit_num; ++i) {
                    auto info = LAPACK_SVD(A_iter, S_iter, U_iter, V_iter,
                                           jobu, jobvt, m, n,
                                           lda, ldu, ldvt,
                                           work.get(), work_size);
                    if (info != 0) {
                        Matrix<T>({m, n}, A_iter).debug();
                        throw_warning("svd works badly error code:" + to_string(info));
                    }

                    A_iter += A_unit_size;
                    S_iter += S_unit_size;
                    if (compute_uv) {
                        U_iter += U_unit_size;
                        V_iter += V_unit_size;
                    }
                }

                if (compute_uv) {
                    return {U, S, V};
                } else {
                    return {S};
                }
            }

            template<typename T>
            auto do_svd(const Matrix<complex<T>> &A, const bool &compute_uv, const bool &full_matrices) -> vector<Matrix<complex<T>>> {
                auto m = static_cast<lapack_int>(A.getShape()[0]), n = static_cast<lapack_int>(A.getShape()[1]);

                auto &k = std::min(m, n);
                mshape_type S_shape = cut(A.getShape(), 1);
                S_shape[0] = k;
                auto S = zeros<T>(S_shape);

                char jobu, jobvt;
                lapack_int lda = m, ldu, ldvt;
                Matrix<complex<T>> U, V;
                svd_prepare_uv(A, compute_uv, full_matrices, m, n, k, jobu, jobvt, ldu, ldvt, U, V);

                auto work_size = LAPACK_SVD_WORK_SIZE<complex<T>>(jobu, jobvt, m, n, lda, ldu, ldvt);
                unique_ptr<complex<T>[]> work(new complex<T>[work_size]);
                unique_ptr<T[]> rwork(new T[5 * k]);

                auto unit_num = A.getUnitNum();
                auto A_unit_size = A.getUnitSize();
                auto S_unit_size = k;
                auto U_unit_size = U.getUnitSize();
                auto V_unit_size = V.getUnitSize();

                auto A_iter = A.begin();
                auto S_iter = S.begin();
                auto U_iter = U.begin();
                auto V_iter = V.begin();

                for (decltype(unit_num) i = 0; i < unit_num; ++i) {
                    auto info = LAPACK_SVD(A_iter, S_iter, U_iter, V_iter,
                                           jobu, jobvt, m, n,
                                           lda, ldu, ldvt,
                                           work.get(), work_size, rwork.get());
                    if (info != 0) {
                        Matrix<complex<T>>({m, n}, A_iter).debug();
                        throw_warning("svd works badly error code:" + to_string(info));
                    }

                    A_iter += A_unit_size;
                    S_iter += S_unit_size;
                    if (compute_uv) {
                        U_iter += U_unit_size;
                        V_iter += V_unit_size;
                    }
                }

                // S perform T to complex<T>
                if (compute_uv) {
                    return {U, S, V};
                } else {
                    return {S};
                }
            }
        }

        template<typename TargetType = void, Calculable dType>
        auto svd(const Matrix<dType> &A, const bool &compute_uv = true, const bool &full_matrices = true)
                -> vector<Matrix<resolve_t<TargetType, linalg_promote_t<dType>>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "svd: output type must be Calculable");
            if (A.getDims() < 2) {
                throw_value_error(
                        "svd: svd requires Matrix at least 2 dimensions, but got: " + to_string(A.getDims()));
            }
            if (A.getShape()[0] > std::numeric_limits<lapack_int>::max() ||
                A.getShape()[1] > std::numeric_limits<lapack_int>::max()) {
                throw_value_error("svd: input Matrix first 2 dimensions exceed the max limit of lapack_int (" +
                                  getCalculableString<lapack_int>() + ") max limits = " +
                                  to_string(std::numeric_limits<lapack_int>::max()) +
                                  " but got input matrix shape = " +
                                  A.shapeToString());
            }
            return Internal::do_svd(Matrix<Out>(A), compute_uv, full_matrices);
        }

        template<typename dType>
        auto rank(const Matrix<dType> &m, float64 tol) -> msize_type {
            // Singular values are non-negative reals, but for complex `m` svd stores
            // them in a complex container; take the magnitude so the result is a real
            // Matrix<float64> (identity for real input).
            Matrix<float64> s = matsar::apply_function(matsar::Math::svd(m, false, false)[0],
                                                       [](const auto &x) { return static_cast<float64>(matsar::abs(x)); });
            Matrix<msize_type> mask = matsar::apply_function<bool, msize_type>(
                    s > tol, [](const bool &x) { return static_cast<msize_type>(x); });
            return msize_type(matsar::Math::sum<msize_type>(mask)(0));
        }

        template<typename dType>
        auto rank(const Matrix<dType> &m) -> msize_type {
            // Singular values are non-negative reals, but for complex `m` svd stores
            // them in a complex container; take the magnitude so the result is a real
            // Matrix<float64> (identity for real input). This also unblocks inv()'s
            // info>0 branch (which calls rank(m)) for complex matrices.
            Matrix<float64> s = matsar::apply_function(matsar::Math::svd(m, false, false)[0],
                                                       [](const auto &x) { return static_cast<float64>(matsar::abs(x)); });
            // TODO: Check the integrity
            auto tol = m.getLength() * matsar::Util::TypeUtil::eps(s.Max());
            Matrix<msize_type> mask = matsar::apply_function<bool, msize_type>(
                    s > tol, [](const bool &x) { return static_cast<msize_type>(x); });
            return msize_type(matsar::Math::sum<msize_type>(mask)(0));
        }
    }
#endif // MATSAR_WITH_BLAS

    namespace Math {
        namespace Internal {
            /**
             * Geometric mean of the n values at(0) .. at(n - 1). The product is kept as
             * mantissa * 2^e, so it neither overflows nor underflows, and its n-th root is
             * 2^q * mantissa^(1/n) * 2^(r/n) with e = q * n + r, |r| < n. NaN for a NaN or
             * negative value (no real geometric mean) and for 0 * Inf; 0 if a value is 0;
             * Inf if one is Inf.
             */
            template<typename At>
            auto geomean_of(msize_type n, At at) -> real_max {
                bool zero = false, inf = false;
                real_max mant = 1;
                long long e = 0;
                for (msize_type k = 0; k < n; ++k) {
                    const auto x = static_cast<real_max>(at(k));
                    if (std::isnan(x) || x < 0)
                        return std::numeric_limits<real_max>::quiet_NaN();
                    if (x == 0) {
                        zero = true;
                        continue;
                    }
                    if (std::isinf(x)) {
                        inf = true;
                        continue;
                    }
                    int ex, em;
                    const real_max mx = std::frexp(x, &ex);   // both factors in [0.5, 1): no underflow
                    mant = std::frexp(mant * mx, &em);
                    e += ex + em;
                }
                if (zero)
                    return inf ? std::numeric_limits<real_max>::quiet_NaN() : real_max(0);
                if (inf)
                    return std::numeric_limits<real_max>::infinity();
                const auto nn = static_cast<long long>(n);
                const long long q = e / nn, r = e - q * nn;
                return std::ldexp(std::pow(mant, real_max(1) / real_max(nn)) * std::exp2(real_max(r) / real_max(nn)),
                                  static_cast<int>(q));
            }
        }

        /**
         * geometric mean, the n-th root of the product of n numbers (computed without overflow).
         * A negative or NaN value makes its mean NaN (there is no real geometric mean).
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         */
        template<typename TargetType = void, RealType dType>
        auto geomean(const Matrix<dType> &m, msize_type dim = 1)
                -> Matrix<resolve_t<TargetType, real_promote_t<dType>>> {
            using Out = resolve_t<TargetType, real_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "geomean: output type must be Calculable");
            msize_type dim_idx = dim - 1;
            if (m.isEmpty())
                return {};
            if (dim_idx >= m.getDims())
                return Matrix<Out>(m);
            const dType *src = m.getData();
            if (dim_idx < 0)
                return Matrix<Out>(mshape_type{1},
                                   Out(Internal::geomean_of(m.getSize(), [src](msize_type k) { return src[k]; })));
            auto shape = m.getShape();
            msize_type step_size = matsar::Util::VectorUtil::mul(shape, 0, dim_idx);
            msize_type unit_size = step_size * shape[dim_idx];
            msize_type step_num = shape[dim_idx];
            msize_type unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<Out> r(r_shape, Out(0));
            Out *dst = r.getData();
            for (msize_type i = 0; i < unit_num; ++i)
                for (msize_type k = 0; k < step_size; ++k) {
                    const dType *base = src + i * unit_size + k;   // the slice: base[j * step_size]
                    dst[i * step_size + k] = Out(Internal::geomean_of(
                            step_num, [base, step_size](msize_type j) { return base[j * step_size]; }));
                }
            return r;
        }

        /**
         * harmonic mean
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         */
        template<typename TargetType = void, RealType dType>
        auto harmmean(const Matrix<dType> &m, msize_type dim = 1)
                -> Matrix<resolve_t<TargetType, real_promote_t<dType>>> {
            using Out = resolve_t<TargetType, real_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "harmmean: output type must be Calculable");
            auto dim_idx = dim - 1;
            if (m.isEmpty())
                return {};
            if (dim_idx >= m.getDims())
                return Matrix<Out>(m);
            if (dim_idx < 0) {
                Out sum = 0.;
                for (auto &i: m)
                    sum += 1. / i;
                return {m.getSize() / sum};
            }
            auto shape = m.getShape();
            auto step_size = matsar::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<Out> r(r_shape, Out(0));   // the sum of reciprocals starts at 0

            auto iter_r = r.getData();
            auto iter_m = m.getData();
            for (msize_type i = 0; i < unit_num; ++i) {
                for (msize_type j = 0; j < step_num; ++j) {
                    for (msize_type k = 0; k < step_size; ++k) {
                        *(iter_r + k) += 1. / (*(iter_m + k));
                    }
                    iter_m += step_size;
                }
                for (msize_type k = 0; k < step_size; ++k) {
                    auto &item = *(iter_r + k);
                    item = (Out) step_num / item;
                }
                iter_r += step_size;
            }
            return r;
        }

        template<typename TargetType = void, Calculable dType>
        auto trace(const Matrix<dType> &m) -> resolve_t<TargetType, accumulate_t<dType>> {
            using Out = resolve_t<TargetType, accumulate_t<dType>>;
            static_assert(calculable_type<Out>::value, "trace: output type must be Calculable");
            if (!m.isSquare() || m.getDims() > 2)
                throw_value_error("trace requires the matrix to be square with 2 dimensions");
            Out r = 0;
            for (msize_type i = 0; i < m.getShape()[0]; ++i)
                r += m(i, i);
            return r;
        }
    }

#ifdef MATSAR_WITH_BLAS
///LU decomposition
    namespace Math {
        inline auto LAPACK_LU(lapack_int m, lapack_int n, float64 *data, lapack_int lda, lapack_int *ipiv) -> lapack_int {
            return LAPACKE_dgetrf2(CblasColMajor, m, n, data, lda, ipiv);
        }

        inline auto LAPACK_LU(lapack_int m, lapack_int n, complex64 *data, lapack_int lda, lapack_int *ipiv) -> lapack_int {
            return LAPACKE_cgetrf2(CblasColMajor, m, n, data, lda, ipiv);
        }

        inline auto LAPACK_LU(lapack_int m, lapack_int n, complex128 *data, lapack_int lda, lapack_int *ipiv) -> lapack_int {
            return LAPACKE_zgetrf2(CblasColMajor, m, n, data, lda, ipiv);
        }

        namespace Internal {
            /// An LU factorization P * A = L * U by LAPACK getrf (partial pivoting).
            template<typename Compute>
            struct LUFactor {
                Matrix<Compute> lu;         // L below the diagonal (unit diagonal implied), U on and above it
                msize_type m = 0, n = 0;
                vector<msize_type> perm;    // row perm[i] of A is row i of P * A
                bool odd = false;           // an odd number of row interchanges: det(P) = -1
            };

            template<typename Compute, typename dType>
            auto lu_factor(const Matrix<dType> &M, const char *name) -> LUFactor<Compute> {
                LUFactor<Compute> f;
                f.lu = Matrix<Compute>(M);
                f.lu.expandShape(2);
                if (!f.lu.is2D())
                    throw_value_error(std::string(name) + " requires a 2D matrix or a matrix that can be expanded to 2D");
                f.m = f.lu.getShape()[0];
                f.n = f.lu.getShape()[1];
                if (f.m > std::numeric_limits<lapack_int>::max() || f.n > std::numeric_limits<lapack_int>::max())
                    throw_value_error(std::string(name) + ": the matrix exceeds the LAPACK index range");
                const msize_type k = std::min(f.m, f.n);
                f.perm.resize(f.m);
                std::iota(f.perm.begin(), f.perm.end(), msize_type(0));
                if (k == 0)
                    return f;
                // getrf fills only min(m, n) pivots: row i was interchanged with row ipiv[i] (1-based)
                vector<lapack_int> ipiv(k);
                LAPACK_LU(lapack_int(f.m), lapack_int(f.n), f.lu.getData(), lapack_int(f.m), ipiv.data());
                for (msize_type i = 0; i < k; ++i)
                    if (ipiv[i] - 1 != i) {
                        std::swap(f.perm[i], f.perm[ipiv[i] - 1]);
                        f.odd = !f.odd;
                    }
                return f;
            }

            /// The unit lower-trapezoidal L (m x min(m, n)) and upper-trapezoidal U (min(m, n) x n).
            template<typename Out, typename Compute>
            void lu_unpack(const LUFactor<Compute> &f, Matrix<Out> &L, Matrix<Out> &U) {
                const msize_type k = std::min(f.m, f.n);
                L = Matrix<Out>(mshape_type{f.m, k}, Out(0));
                U = Matrix<Out>(mshape_type{k, f.n}, Out(0));
                for (msize_type j = 0; j < k; ++j) {
                    L(j, j) = Out(1);
                    for (msize_type i = j + 1; i < f.m; ++i)
                        L(i, j) = Out(f.lu(i, j));
                }
                for (msize_type i = 0; i < k; ++i)
                    for (msize_type j = i; j < f.n; ++j)
                        U(i, j) = Out(f.lu(i, j));
            }
        }

        /**
         * [L, U] = lu(A) as MATLAB: A = L * U with L = P' * (unit lower triangular), a row
         * permutation of a lower-triangular matrix. A is m x n: L is m x min(m, n), U is min(m, n) x n.
         */
        template<typename TargetType = void, Calculable dType>
        auto lu(const Matrix<dType> &M)
                -> tuple<Matrix<resolve_t<TargetType, linalg_promote_t<dType>>>, Matrix<resolve_t<TargetType, linalg_promote_t<dType>>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "lu: output type must be Calculable");
            auto f = Internal::lu_factor<linalg_promote_t<dType>>(M, "lu");
            Matrix<Out> L, U;
            Internal::lu_unpack(f, L, U);
            const msize_type k = std::min(f.m, f.n);
            if (k == 0)
                return {L, U};
            Matrix<Out> PL(L.getShape(), Out(0));
            for (msize_type i = 0; i < f.m; ++i)
                for (msize_type j = 0; j < k; ++j)
                    PL(f.perm[i], j) = L(i, j);
            return {PL, U};
        }

        /**
         * [L, U, P] = lu(A) as MATLAB: P * A = L * U with a unit lower-triangular L (m x min(m, n)),
         * an upper-triangular U (min(m, n) x n) and the m x m permutation matrix P.
         */
        template<typename TargetType = void, Calculable dType>
        auto lup(const Matrix<dType> &M)
                -> tuple<Matrix<resolve_t<TargetType, linalg_promote_t<dType>>>, Matrix<resolve_t<TargetType, linalg_promote_t<dType>>>, Matrix<int64>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "lup: output type must be Calculable");
            auto f = Internal::lu_factor<linalg_promote_t<dType>>(M, "lup");
            Matrix<Out> L, U;
            Internal::lu_unpack(f, L, U);
            Matrix<int64> P(mshape_type{f.m, f.m}, int64(0));
            for (msize_type i = 0; i < f.m; ++i)
                P(i, f.perm[i]) = 1;
            return {L, U, P};
        }
    }

    namespace Math {
        template<typename TargetType = void, Calculable dType>
        auto det(const Matrix<dType> &m) -> Matrix<resolve_t<TargetType, linalg_promote_t<dType>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "det: output type must be Calculable");
            using Compute = linalg_promote_t<dType>;
            if (m.isEmpty())   // MATLAB: det([]) = 1
                return Matrix<Out>(mshape_type{1}, Out(1));
            Matrix<dType> a(m);
            a.expandShape(2);
            if (!a.is2D() || a.getShape()[0] != a.getShape()[1])
                throw_value_error("det: requires a square matrix, but got shape " + m.shapeToString());
            // P * A = L * U with a unit-diagonal L: det(A) = det(P) * prod(diag(U)), where det(P)
            // is -1 for an odd number of row interchanges (det([[4,3],[6,3]]) = -6).
            auto f = Internal::lu_factor<Compute>(a, "det");
            Compute d = f.odd ? Compute(-1) : Compute(1);
            for (msize_type i = 0; i < f.n; ++i)
                d *= f.lu(i, i);
            return Matrix<Out>(mshape_type{1}, Out(d));
        }
    }

    namespace Math {
        template<typename TargetType = void, Calculable dType>
        auto inv(const Matrix<dType> &m) -> Matrix<resolve_t<TargetType, linalg_promote_t<dType>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "inv: output type must be Calculable");
            using Compute = linalg_promote_t<dType>;
            if (m.isEmpty())   // MATLAB: inv([]) = []
                return {};
            if (!m.isSquare() || m.getDims() != 2)
                throw_value_error("inv: requires a 2-D square matrix, but got shape " + m.shapeToString());

            auto n = m.getShape()[0];
            Matrix<Compute> A(m), X({n, n}, 0., m.getData());
            lapack_int iter, info;
            unique_ptr<lapack_int[]> IPIV(new lapack_int[n]);

            if constexpr (matsar::complex_type<dType>::value)      // <- the ONLY real/complex difference
                info = LAPACKE_zcgesv(CblasColMajor, n, n, A.getData(), n, IPIV.get(),
                                      eye<Compute>(n).getData(), n, X.getData(), n, &iter);
            else
                info = LAPACKE_dsgesv(CblasColMajor, n, n, A.getData(), n, IPIV.get(),
                                      eye<Compute>(n).getData(), n, X.getData(), n, &iter);

            if (info < 0)
                throw_value_error("inv argument error " + to_string(info));
            if (info > 0) {   // an exactly zero pivot: singular. MATLAB warns and returns Inf.
                stringstream ss;
                ss << "inv: the matrix is singular (rank " << rank(m) << " of " << n << "); the result is Inf";
                throw_warning(ss.str());
                return Matrix<Out>(mshape_type{n, n}, Out(std::numeric_limits<real_max>::infinity()));
            }
            return X;
        }
    }
    namespace Math {
//        /**
//         * @author wzp
//         * @function norm norm
//         * @param n = 0 represents the infinity norm
//         * @param n = 1 represents the 1-norm
//         * @param n = 2 the 2-norm
//         *
//         */
//        template<typename dType>
//        auto norm(Matrix<dType> m, int n = 2) -> dType {
//            if (!m.is2D())
//                throw_value_error("not a 2D array!");
//            if (n == 2) {
//                return matsar::Math::sqrt(matsar::Math::svd(m * m.T())).Max();
//            } else if (n == 1) {
//                auto sp = m.getShape();
//                m = matsar::Math::cumsum(matsar::Math::abs(m), 2);
//                dType max_val = 0;
//                for (int i = 0; i < sp[0]; i++) {
//                    if (max_val < m(i, sp[1] - 1))
//                        max_val = m(i, sp[1] - 1);
//                }
//                return max_val;
//            } else if (n == 0) {
//                m = matsar::Math::cumsum(matsar::Math::abs(m.T()), 2);
//                auto sp = m.getShape();
//                dType max_val = 0;
//                for (int i = 0; i < sp[0]; i++) {
//                    if (max_val < m(i, sp[1] - 1))
//                        max_val = m(i, sp[1] - 1);
//                }
//                return max_val;
//            } else {
//                throw_value_error("p-norm only supports 1, 2, ∞!");
//                return 0;
//            }
//        }

        /**
        * @author wzp
        * @fuction [V,D, W] = eig(A)
        * @return returns the diagonal matrix D of eigenvalues and the matrix V, whose columns are the corresponding right eigenvectors, such that A*V = V*D.
        * @return also returns the full matrix W, whose columns are the corresponding left eigenvectors, such that W'*A = D*W'*B.
        */
        template<typename TargetType = void, Calculable dType>
        auto eig(const Matrix<dType> &A) -> vector<Matrix<resolve_t<TargetType, linalg_promote_t<dType>>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "eig: output type must be Calculable");
            constexpr bool complex_out = complex_type<Out>::value;
            if constexpr (complex_type<dType>::value) {   // a complex matrix: LAPACK zgeev
                static_assert(complex_out, "eig: a complex matrix needs a complex result type");
                Matrix<complex128> cA(A);
                cA.expandShape(2);
                if (!cA.is2D())
                    throw_value_error("eig requires a 2D matrix or a matrix that can be expanded to 2D");
                const lapack_int n = cA.getShape()[0];
                if (cA.getShape()[1] != n)
                    throw_value_error("eig requires a square matrix");
                vector<complex128> w(n), vl(n * n), vr(n * n);
                auto info = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'V', 'V', n, cA.getData(), n, w.data(),
                                          vl.data(), n, vr.data(), n);
                if (info != 0)
                    throw_warning("eig works badly error code:" + to_string(info));
                Matrix<Out> V(mshape_type{n, n}, Out(0)), D(mshape_type{n, n}, Out(0)), W(mshape_type{n, n}, Out(0));
                for (lapack_int c = 0; c < n; ++c) {
                    for (lapack_int r = 0; r < n; ++r) {
                        V(r, c) = Out(vr[r + c * n]);   // right eigenvectors: A*V = V*D
                        W(r, c) = Out(vl[r + c * n]);   // left eigenvectors:  W'*A = D*W'
                    }
                    D(c, c) = Out(w[c]);
                }
                return {V, D, W};
            }
            // a real matrix: LAPACK dgeev
            Matrix<float64> fA(A.real());
            fA.expandShape(2);
            if (!fA.is2D())
                throw_value_error("eig requires a 2D matrix or a matrix that can be expanded to 2D");
            lapack_int m = fA.getShape()[0], n = fA.getShape()[1];
            if (m != n)
                throw_value_error("eig requires a square matrix");
            unique_ptr<float64[]> wr(new float64[m]);
            unique_ptr<float64[]> wi(new float64[m]);
            unique_ptr<float64[]> vl(new float64[m * m]);
            unique_ptr<float64[]> vr(new float64[m * m]);
            auto info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'V', 'V', m, fA.getData(), m, wr.get(), wi.get(),
                                      vl.get(), m, vr.get(), m);
            if (info != 0) {
                A.debug();
                throw_warning("eig works badly error code:" + to_string(info));
            }
            if constexpr (!complex_out) {
                for (lapack_int j = 0; j < m; ++j)
                    if (wi[j] != 0)
                        throw_value_error("eig: the matrix has complex eigenvalues; request a complex result, "
                                          "e.g. eig<complex128>(A)");
            }
            // Column c of dgeev's VR / VL (column-major, leading dimension m). A complex-conjugate
            // eigenvalue pair (wi[c] > 0, wi[c+1] < 0) is stored as two real columns:
            // x_c = X(:,c) + i*X(:,c+1) and x_{c+1} = conj(x_c).
            auto entry = [&](const float64 *X, lapack_int r, lapack_int c) -> complex128 {
                const float64 x = X[r + c * m];
                if (wi[c] > 0) return {x, X[r + (c + 1) * m]};
                if (wi[c] < 0) return {X[r + (c - 1) * m], -x};
                return {x, 0};
            };
            Matrix<Out> V(mshape_type{m, m}, Out(0)), D(mshape_type{m, m}, Out(0)), W(mshape_type{m, m}, Out(0));
            for (lapack_int c = 0; c < m; ++c) {
                for (lapack_int r = 0; r < m; ++r) {
                    if constexpr (complex_out) {
                        V(r, c) = Out(entry(vr.get(), r, c));   // right eigenvectors: A*V = V*D
                        W(r, c) = Out(entry(vl.get(), r, c));   // left eigenvectors:  W'*A = D*W'
                    } else {
                        V(r, c) = Out(vr[r + c * m]);
                        W(r, c) = Out(vl[r + c * m]);
                    }
                }
                if constexpr (complex_out)
                    D(c, c) = Out(wr[c], wi[c]);
                else
                    D(c, c) = Out(wr[c]);
            }
            return {V, D, W};
        }

        /**
         * @author wzp
         * @function condeig condition numbers related to the eigenvalues
         * @return [V,D,s] = condeig(A)
         * s = a vector of condition numbers for the eigenvalues of A. These condition numbers are the reciprocals of the cosines of the angles between the left and right eigenvectors.
         */
        template<typename TargetType = void, Calculable dType>
        auto condeig(const Matrix<dType> &A) -> vector<Matrix<resolve_t<TargetType, linalg_promote_t<dType>>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "condeig: output type must be Calculable");
            if (!A.is2D() || !A.isSquare())
                throw_value_error("condeig: requires a square matrix, but got shape " + A.shapeToString());
            auto V_D_W = eig<complex128>(A);
            const auto &V = V_D_W[0], &D = V_D_W[1], &W = V_D_W[2];
            const msize_type n = D.getShape()[0];
            if constexpr (!complex_type<Out>::value) {
                for (msize_type j = 0; j < n; ++j)
                    if (D(j, j).imag() != 0)
                        throw_value_error("condeig: the matrix has complex eigenvalues; request a complex result, "
                                          "e.g. condeig<complex128>(A)");
            }
            // to Out; for a real Out the spectrum is real, and so are its eigenvectors
            auto to_out = [](const complex128 &z) -> Out {
                if constexpr (complex_type<Out>::value)
                    return Out(z);
                else
                    return Out(z.real());
            };
            // eigenvectors are the COLUMNS of V (right) and W (left): s_j = |w_j| |v_j| / |w_j^H v_j|
            Matrix<Out> Vo(mshape_type{n, n}, Out(0)), Do(mshape_type{n, n}, Out(0)), s(mshape_type{n}, Out(0));
            for (msize_type j = 0; j < n; ++j) {
                complex128 dot = 0;
                real_max nv = 0, nw = 0;
                for (msize_type i = 0; i < n; ++i) {
                    dot += std::conj(W(i, j)) * V(i, j);
                    nv += std::norm(V(i, j));
                    nw += std::norm(W(i, j));
                    Vo(i, j) = to_out(V(i, j));
                }
                Do(j, j) = to_out(D(j, j));
                s(j) = Out(std::sqrt(nv) * std::sqrt(nw) / std::abs(dot));
            }
            return {Vo, Do, s};
        }

        /**
         * Moore-Penrose pseudo-inverse from the SVD, as MATLAB pinv(A): with A = U S V',
         * pinv(A) = V S^+ U' where S^+ inverts the singular values above
         * tol = max(m, n) * eps(max(S)) and drops the rest, so a rank-deficient A works.
         * A vector is a column (n x 1); the result is n x m for an m x n A.
         */
        template<typename TargetType = void, Calculable dType>
        auto pinv(const Matrix<dType> &matrix) -> Matrix<resolve_t<TargetType, linalg_promote_t<dType>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "pinv: output type must be Calculable");
            using Compute = linalg_promote_t<dType>;
            if (matrix.isEmpty())
                return {};
            Matrix<Compute> A(matrix);
            A.expandShape(2);
            if (!A.is2D())
                throw_value_error("pinv: requires a 2-D matrix, but got shape " + matrix.shapeToString());
            const msize_type m = A.getShape()[0], n = A.getShape()[1], k = std::min(m, n);
            auto usv = svd(A, true, false);   // economy: U is m x k, S has k values, Vh is k x n
            const auto &U = usv[0], &S = usv[1], &Vh = usv[2];
            real_max smax = 0;
            for (msize_type l = 0; l < k; ++l)
                smax = std::max<real_max>(smax, std::abs(S(l)));
            const real_max tol = real_max(std::max(m, n)) * matsar::Util::TypeUtil::eps(smax);
            auto cj = [](const Compute &x) -> Compute {
                if constexpr (complex_type<Compute>::value)
                    return std::conj(x);
                else
                    return x;
            };
            Matrix<Compute> X(mshape_type{n, m}, Compute(0));
            for (msize_type l = 0; l < k; ++l) {
                const real_max s = std::abs(S(l));
                if (!(s > tol))
                    continue;
                for (msize_type j = 0; j < m; ++j) {
                    const Compute u = cj(U(j, l)) / s;
                    for (msize_type i = 0; i < n; ++i)
                        X(i, j) += cj(Vh(l, i)) * u;
                }
            }
            return Matrix<Out>(X);
        }

        namespace Internal {
            /**
             * Schur decomposition A = U T U' by LAPACK ?gees, in place: T holds A on entry.
             * C = float64 gives the real Schur form (T quasi-upper-triangular: 2x2 diagonal blocks
             * [[a, b], [c, a]] with b c < 0 for complex conjugate eigenvalue pairs), C = complex128
             * the complex form (T upper triangular). w receives the eigenvalues.
             */
            template<typename C>
            void schur_factor(Matrix<C> &T, Matrix<C> &U, vector<complex128> &w) {
                w.clear();
                U = Matrix<C>();
                if (T.isEmpty())
                    return;
                T.expandShape(2);
                if (!T.is2D() || T.getShape()[0] != T.getShape()[1])
                    throw_value_error("schur: requires a square matrix, but got shape " + T.shapeToString());
                const lapack_int n = T.getShape()[0];
                U = Matrix<C>(mshape_type{n, n}, C(0));
                lapack_int sdim = 0, info;
                if constexpr (complex_type<C>::value) {
                    w.resize(n);
                    info = LAPACKE_zgees(LAPACK_COL_MAJOR, 'V', 'N', nullptr, n, T.getData(), n, &sdim, w.data(),
                                         U.getData(), n);
                } else {
                    vector<float64> wr(n), wi(n);
                    info = LAPACKE_dgees(LAPACK_COL_MAJOR, 'V', 'N', nullptr, n, T.getData(), n, &sdim, wr.data(),
                                         wi.data(), U.getData(), n);
                    for (lapack_int j = 0; j < n; ++j)
                        w.emplace_back(wr[j], wi[j]);
                }
                if (info != 0)
                    throw_warning("schur: the QR algorithm did not converge (LAPACK info " + to_string(info) + ")");
            }
        }

        /**
         * T = schur(A, flag) as MATLAB. "real": the real Schur form of a real A (quasi-triangular,
         * a 2x2 block per complex conjugate eigenvalue pair). "complex": the triangular complex
         * Schur form; it is complex when A has complex eigenvalues, so request a complex result,
         * e.g. schur<complex128>(A, "complex"). A complex A always gives the complex form.
         */
        template<typename TargetType = void, Calculable dType>
        auto schur(const Matrix<dType> &A, const std::string &flag) -> Matrix<resolve_t<TargetType, linalg_promote_t<dType>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "schur: output type must be Calculable");
            if (flag != "real" && flag != "complex")
                throw_value_error("schur: the flag must be 'real' or 'complex', but got '" + flag + "'");
            vector<complex128> w;
            if constexpr (complex_type<dType>::value || complex_type<Out>::value) {
                if (complex_type<dType>::value || flag == "complex") {
                    Matrix<complex128> T(A), U;
                    Internal::schur_factor(T, U, w);
                    return Matrix<Out>(T);
                }
            }
            Matrix<float64> T(A.real()), U;   // (a complex A returned above)
            Internal::schur_factor(T, U, w);
            if (flag == "complex")   // a real result type: only a real spectrum has a real triangular form
                for (const auto &z: w)
                    if (z.imag() != 0)
                        throw_value_error("schur: the matrix has complex eigenvalues, so its complex Schur form is "
                                          "complex; request a complex result, e.g. schur<complex128>(A, \"complex\")");
            return Matrix<Out>(T);
        }

        /**
         * [U, T] = schur(A) as MATLAB: A = U * T * U' with U orthogonal (unitary for a complex A)
         * and T the real Schur form of a real A, or the triangular complex form of a complex A.
         * Returns {U, T}.
         */
        template<typename TargetType = void, Calculable dType>
        auto schur(const Matrix<dType> &A) -> vector<Matrix<resolve_t<TargetType, linalg_promote_t<dType>>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "schur: output type must be Calculable");
            using C = std::conditional_t<complex_type<dType>::value, complex128, float64>;
            Matrix<C> T(A), U;
            vector<complex128> w;
            Internal::schur_factor(T, U, w);
            return {Matrix<Out>(U), Matrix<Out>(T)};
        }

        /**
         * MATLAB polyvalm(p, X): the polynomial p (highest power first) of the square matrix X,
         * by Horner's rule Y = X Y + p_i I.
         */
        template<typename dType>
        auto polyvalm(const vector<dType> &p, const Matrix<dType> &X) -> Matrix<dType> {
            Matrix<dType> x(X);
            if (x.isEmpty())
                return {};
            x.expandShape(2);
            if (!x.is2D() || x.getShape()[0] != x.getShape()[1])
                throw_value_error("polyvalm: requires a square matrix, but got shape " + X.shapeToString());
            const msize_type n = x.getShape()[0];
            Matrix<dType> Y(mshape_type{n, n}, dType(0));
            for (const auto &c: p) {
                Y = mmul<dType, dType, dType>(x, Y);
                for (msize_type i = 0; i < n; ++i)
                    Y(i, i) += c;
            }
            return Y;
        }

        template<typename dType, typename FT>
        auto funm(const Matrix<dType> &A, FT matrix_fun, const string &fun_name) -> Matrix<dType> {
            vector<Matrix<dType>> V_D_W = eig(A);
            auto inter_m = diag(V_D_W[1]);
            matrix_fun(inter_m, fun_name);
            // V holds the right eigenvectors as columns: f(A) = V f(D) V^-1
            return mmul(mmul(V_D_W[0], diag(inter_m)), inv(V_D_W[0]));
        }
    }
#endif // MATSAR_WITH_BLAS

    namespace Math {
        /// MATLAB polyder(p): the derivative of p (highest power first), without leading zeros.
        template<typename dType>
        auto polyder(const vector<dType> &p) -> vector<dType> {
            if (p.size() <= 1) return {dType(0)};
            vector<dType> r(p.begin(), p.end() - 1);
            auto n = r.size();
            for (auto &i: r)
                i *= dType(n--);
            auto first = std::find_if(r.begin(), r.end() - 1, [](const dType &v) { return v != dType(0); });
            r.erase(r.begin(), first);   // as MATLAB: drop leading zeros, keep at least one coefficient
            return r;
        }

        /**
         * @author zzx
         * MATLAB polyder(a, b): the derivative of the product a * b.
         */
        template<typename dType>
        auto polyder(const vector<dType> &a, const vector<dType> &b) -> vector<dType> {
            if (a.empty() || b.empty())
                return {dType(0)};
            vector<dType> merge(a.size() + b.size() - 1, 0.);
            auto an = a.size() - 1;
            auto bn = b.size() - 1;
            for (int i = 0; i < a.size(); ++i) {
                for (int j = 0; j < b.size(); ++j) {
                    merge[an - i + bn - j] += a[an - i] * b[bn - j];
                }
            }
            return polyder(merge);
        }

        template<typename dType>
        auto polyeval(const vector<dType> &coef, dType x) -> dType {
            if (coef.size() == 0) return 0.;
            dType r = 0.;
            dType x_cal = 1;
            for (int i = 0; i < coef.size(); ++i) {
                r += x_cal * coef[coef.size() - i - 1];
                x_cal *= x;
            }
            return r;
        }

        /**
         * @author zzx
         * MATLAB ppval for the piecewise polynomial (breaks, coefs): coefs has one row per piece
         * (breaks.size() - 1 rows), highest power first; a vector is a single piece. Piece i covers
         * breaks[i] <= x < breaks[i + 1] and is evaluated at x - breaks[i]; x before the first or
         * after the last break uses the first or last piece. NaN gives NaN.
         */
        template<typename dType>
        auto ppval(const vector<dType> &breaks, const Matrix<dType> &coefs, const vector<dType> &xq) -> vector<dType> {
            if (breaks.size() < 2)
                throw_value_error("ppval: at least two breaks are required, but got " + to_string(breaks.size()));
            const auto pieces = static_cast<msize_type>(breaks.size() - 1);
            Matrix<dType> c(coefs);
            if (c.getDims() == 1)
                c = c.reshape(mshape_type{1, c.getSize()});
            if (!c.is2D() || c.getShape()[0] != pieces)
                throw_value_error("ppval: coefs must have one row per piece (" + to_string(pieces) +
                                  " for these breaks), but got shape " + coefs.shapeToString());
            const msize_type order = c.getShape()[1];
            vector<dType> r(xq.size());
            for (std::size_t q = 0; q < xq.size(); ++q) {
                const dType x = xq[q];   // (NaN evaluates to NaN)
                msize_type i = 0;
                while (i < pieces - 1 && !(x < breaks[i + 1]))
                    ++i;
                const dType t = x - breaks[i];
                dType v = 0;
                for (msize_type k = 0; k < order; ++k)   // Horner
                    v = v * t + c(i, k);
                r[q] = v;
            }
            return r;
        }

        namespace Internal {
            // Exponential integral Ei(x) for real x != 0 -- native drop-in for
            // boost::math::expint(x), so this module keeps OpenBLAS as its only
            // external dependency. Matches boost/reference Ei to ~1e-15.
            inline real_max expint_ei(real_max x) {
                constexpr real_max EULER = 0.5772156649015328606;
                constexpr int MAXIT = 200;
                const real_max EPS = std::numeric_limits<real_max>::epsilon();
                const real_max FPMIN = std::numeric_limits<real_max>::min() / EPS;
                if (x == 0.0) return -std::numeric_limits<real_max>::infinity();
                if (x < 0.0) {                        // Ei(x) = -E1(-x); E1 via Lentz continued fraction
                    const real_max y = -x;
                    real_max b = y + 1.0, c = 1.0 / FPMIN, d = 1.0 / b, h = d;
                    for (int i = 1; i <= MAXIT; ++i) {
                        const real_max a = -real_max(i) * i;
                        b += 2.0;
                        d = 1.0 / (a * d + b);
                        c = b + a / c;
                        const real_max del = c * d;
                        h *= del;
                        if (std::fabs(del - 1.0) <= EPS) break;
                    }
                    return -std::exp(-y) * h;
                }
                if (x < FPMIN) return std::log(x) + EULER;
                if (x <= -std::log(EPS)) {            // power series (moderate x)
                    real_max sum = 0.0, fact = 1.0;
                    for (int k = 1; k <= MAXIT; ++k) {
                        fact *= x / k;
                        const real_max term = fact / k;
                        sum += term;
                        if (term < EPS * sum) break;
                    }
                    return sum + std::log(x) + EULER;
                }
                real_max sum = 0.0, term = 1.0;       // asymptotic series (large x)
                for (int k = 1; k <= MAXIT; ++k) {
                    const real_max prev = term;
                    term *= real_max(k) / x;
                    if (term < EPS) break;
                    if (term < prev)
                        sum += term;
                    else {
                        sum -= prev;
                        break;
                    }
                }
                return std::exp(x) * (1.0 + sum) / x;
            }

            /**
             * E1(z) = integral from z to Inf of e^-t / t dt (MATLAB expint), principal branch: on
             * the cut, the negative real axis, the value just above it, E1(-x) = -Ei(x) - i pi.
             * The power series -gamma - log z - sum_k (-z)^k / (k k!) where its terms do not cancel
             * much (|z| + Re z <= 4: the loss is at most e^4), else the continued fraction
             * E1(z) = e^-z / (z + 1 - 1 / (z + 3 - 4 / (z + 5 - ...))) by the modified Lentz method.
             */
            inline auto expint_e1(complex<real_max> z) -> complex<real_max> {
                using C = complex<real_max>;
                constexpr real_max EULER = 0.5772156649015328606;
                constexpr real_max PI = 3.14159265358979323846;
                constexpr real_max NaN = std::numeric_limits<real_max>::quiet_NaN();
                const real_max EPS = std::numeric_limits<real_max>::epsilon();
                const real_max x = z.real(), y = z.imag();
                if (std::isnan(x) || std::isnan(y))
                    return {NaN, NaN};
                if (y == 0) {
                    if (x == 0)
                        return {std::numeric_limits<real_max>::infinity(), 0};
                    if (x < 0)   // -Ei(|x|): a positive series, no cancellation
                        return {-expint_ei(-x), -PI};
                    if (std::isinf(x))
                        return {0, 0};
                }
                const real_max az = std::abs(z);
                if (std::isinf(az))
                    return x > 0 && std::isfinite(y) ? C(0, 0) : C(NaN, NaN);
                if (az + x <= 4 && az <= 600) {
                    C term = 1, sum = 0;
                    for (int k = 1; k <= 20000; ++k) {
                        term *= -z / real_max(k);
                        const C t = term / real_max(k);
                        sum += t;
                        if (std::abs(t) <= EPS * std::abs(sum))
                            break;
                    }
                    return -EULER - std::log(z) - sum;
                }
                const real_max TINY = std::numeric_limits<real_max>::min() / EPS;
                C b = z + real_max(1), c = real_max(1) / TINY, d = real_max(1) / b, h = d;
                for (int i = 1; i <= 100000; ++i) {
                    const real_max a = -real_max(i) * real_max(i);
                    b += real_max(2);
                    C den = a * d + b;
                    if (den == C(0))
                        den = TINY;
                    d = real_max(1) / den;
                    c = b + a / c;
                    if (c == C(0))
                        c = TINY;
                    const C del = c * d;
                    h *= del;
                    if (std::abs(del - real_max(1)) <= EPS)
                        break;
                }
                return h * std::exp(-z);
            }
        }

        /**
         * @author zzx
         * MATLAB expint: the exponential integral E1(x). For a real x < 0, E1 is complex
         * (-Ei(-x) - i pi): a real result type gives NaN there, a complex one the value, e.g.
         * Expint<complex128>(x). A complex argument needs a complex result type.
         */
        template<typename TargetType = void, Calculable dType>
        auto Expint(const Matrix<dType> &x) -> Matrix<resolve_t<TargetType, real_promote_t<dType>>> {
            using Out = resolve_t<TargetType, real_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "Expint: output type must be Calculable");
            static_assert(complex_type<Out>::value || !complex_type<dType>::value,
                          "Expint: a complex argument needs a complex result type");
            Matrix<Out> r(x.getShape(), Out(0));
            const dType *src = x.getData();
            Out *dst = r.getData();
            for (msize_type k = 0; k < x.getSize(); ++k) {
                if constexpr (complex_type<Out>::value) {
                    const complex<real_max> z(static_cast<real_max>(std::real(src[k])),
                                              static_cast<real_max>(std::imag(src[k])));
                    dst[k] = Out(Internal::expint_e1(z));
                } else {
                    const auto v = static_cast<real_max>(src[k]);
                    dst[k] = (std::isnan(v) || v < 0) ? Out(std::numeric_limits<real_max>::quiet_NaN())
                                                      : Out(Internal::expint_e1({v, 0}).real());
                }
            }
            return r;
        }

        namespace Internal {
            /// MATLAB's placement of diagonal k of the m x n matrix A in column c of B
            /// (min(m, n) rows): m >= n: B(j, c) = A(j - k, j); m < n: B(i, c) = A(i, i + k).
            inline void spdiags_place(const Matrix<double> &A, msize_type k, Matrix<double> &B, msize_type c) {
                const msize_type m = A.getShape()[0], n = A.getShape()[1];
                if (m >= n) {
                    for (msize_type j = 0; j < n; ++j)
                        if (j - k >= 0 && j - k < m)
                            B(j, c) = A(j - k, j);
                } else {
                    for (msize_type i = 0; i < m; ++i)
                        if (i + k >= 0 && i + k < n)
                            B(i, c) = A(i, i + k);
                }
            }

            inline auto spdiags_input(const Matrix<double> &A) -> Matrix<double> {
                Matrix<double> a(A);
                a.expandShape(2);
                if (!a.is2D())
                    throw_value_error("spdiags: requires a 2-D matrix, but got shape " + A.shapeToString());
                return a;
            }
        }

        /**
         * [B, d] = spdiags(A) as MATLAB: d lists the diagonals of A holding a nonzero, in ascending
         * order (diagonal k holds A(i, i + k)), and column c of B (min(m, n) rows) holds diagonal
         * d(c), placed as MATLAB does (Internal::spdiags_place). Empty results for an all-zero A.
         */
        inline auto spdiags(const Matrix<double> &A) -> tuple<Matrix<double>, Matrix<double>> {
            if (A.isEmpty())
                return {};
            const auto a = Internal::spdiags_input(A);
            const msize_type m = a.getShape()[0], n = a.getShape()[1];
            vector<double> d;
            for (msize_type k = -(m - 1); k < n; ++k)
                for (msize_type i = std::max<msize_type>(0, -k); i < m && i + k < n; ++i)
                    if (a(i, i + k) != 0) {
                        d.push_back(double(k));
                        break;
                    }
            if (d.empty())
                return {};
            Matrix<double> B(mshape_type{std::min(m, n), msize_type(d.size())}, 0.);
            for (msize_type c = 0; c < msize_type(d.size()); ++c)
                Internal::spdiags_place(a, msize_type(d[c]), B, c);
            return {B, Matrix<double>(d)};
        }

        /// B = spdiags(A, d) as MATLAB: column c of B holds diagonal d(c) of A (zeros where that
        /// diagonal is outside A), placed as in [B, d] = spdiags(A).
        inline auto spdiags(const Matrix<double> &A, const Matrix<double> &d) -> Matrix<double> {
            if (A.isEmpty() || d.isEmpty())
                return {};
            const auto a = Internal::spdiags_input(A);
            const msize_type m = a.getShape()[0], n = a.getShape()[1];
            Matrix<double> B(mshape_type{std::min(m, n), d.getSize()}, 0.);
            const double *dk = d.getData();
            for (msize_type c = 0; c < d.getSize(); ++c)
                Internal::spdiags_place(a, msize_type(dk[c]), B, c);
            return B;
        }
    }
}

#endif //MATSAR_LINEARALGEBRA_HPP
