// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2022/7/9.
//
///LA Linear Algebra
#ifndef MATSAR_LINEARALGEBRA_HPP
#define MATSAR_LINEARALGEBRA_HPP

#include <string>

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

    namespace Math {
        namespace Internal {
            auto LAPACK_SVD(double *a, double *s, double *u, double *vt,
                            char jobu, char jobvt,
                            lapack_int m, lapack_int n,
                            lapack_int lda,
                            lapack_int ldu, lapack_int ldvt,
                            double *work, lapack_int lwork) -> lapack_int {
                return LAPACKE_dgesvd_work(CblasColMajor, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
            }

            auto LAPACK_SVD(lapack_complex_float *a, float *s, lapack_complex_float *u, lapack_complex_float *vt,
                            char jobu, char jobvt,
                            lapack_int m, lapack_int n,
                            lapack_int lda,
                            lapack_int ldu,
                            lapack_int ldvt, lapack_complex_float *work,
                            lapack_int lwork, float *rwork) -> lapack_int {
                return LAPACKE_cgesvd_work(CblasColMajor, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
                                           rwork);
            }

            auto LAPACK_SVD(lapack_complex_double *a, double *s, lapack_complex_double *u, lapack_complex_double *vt,
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
            auto LAPACK_SVD_WORK_SIZE<float64>(char jobu, char jobvt,
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
            auto LAPACK_SVD_WORK_SIZE<complex64>(char jobu, char jobvt,
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
            auto LAPACK_SVD_WORK_SIZE<complex128>(char jobu, char jobvt,
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

            template<typename T>
            void svd_prepare_uv(const Matrix<T> &A, const bool &compute_uv, const bool &full_matrices, int m, int n,
                                const int &k,
                                char &jobu, char &jobvt, int &ldu, int &ldvt, Matrix<T> &U, Matrix<T> &V) {
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
                    U = zeros<T>(U_shape);
                    V = zeros<T>(V_shape);
                } else {
                    jobu = 'N';
                    jobvt = 'N';
                    ldu = 1;
                    ldvt = 1;
                }
            }

            auto do_svd(const Matrix<float64> &A, const bool &compute_uv, const bool &full_matrices) -> vector<Matrix<float64>> {
                using T = float64;

                auto m = static_cast<lapack_int>(A.getShape()[0]), n = static_cast<lapack_int>(A.getShape()[1]);

                auto &k = min(m, n);
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

                auto &k = min(m, n);
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

        template<typename RT>
        auto svd(const Matrix<RT> &A, const bool &compute_uv = true, const bool &full_matrices = true) -> vector<Matrix<float64>> {
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
            return Internal::do_svd(Matrix<float64>(A), compute_uv, full_matrices);
        }

        template<typename T>
        auto svd(const Matrix<complex<T>> &A, const bool &compute_uv = true, const bool &full_matrices = true) -> vector<Matrix<complex<T>>> {
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
            return Internal::do_svd(A, compute_uv, full_matrices);
        }

        template<typename dType>
        auto rank(const Matrix<dType> &m, float64 tol) -> msize_type {
            Matrix<float64> s = matsar::Math::svd(m, false, false)[0];
            return msize_type(matsar::Math::sum(s > tol)(0));
        }

        template<typename dType>
        auto rank(const Matrix<dType> &m) -> msize_type {
            Matrix<float64> s = matsar::Math::svd(m, false, false)[0];
            // TODO: Check the integrity
            auto tol = m.getLength() * matsar::Util::TypeUtil::eps(s.Max());
            return msize_type(matsar::Math::sum(s > tol)(0));
        }
    }

    namespace Math {
        /**
         * geomean computes the product of n numbers, then takes the nth root
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         */
        template<RealType dType>
        auto geomean(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<real_max> {
            msize_type dim_idx = dim - 1;
            if (m.isEmpty())
                return {};
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0) {
                dType mul = 1;
                for (auto &i: m)
                    mul *= i;
                return {pow(mul, 1. / m.getSize())};
            }
            auto shape = m.getShape();
            msize_type step_size = matsar::Util::VectorUtil::mul(shape, 0, dim_idx);
            msize_type unit_size = step_size * shape[dim_idx];
            msize_type step_num = shape[dim_idx];
            msize_type unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<real_max> r(r_shape, 1, m.getData());//the product must be initialized to 1

            auto iter_r = r.getData();
            auto iter_m = m.getData();
            float_max power = 1. / (float_max) step_num;
            for (msize_type i = 0; i < unit_num; ++i) {
                for (msize_type j = 0; j < step_num; ++j) {
                    for (msize_type k = 0; k < step_size; ++k)
                        *(iter_r + k) *= *(iter_m + k);
                    iter_m += step_size;
                }
                for (msize_type k = 0; k < step_size; ++k) {
                    auto &item = *(iter_r + k);
                    item = pow(item, power);
                }
                iter_r += step_size;
            }
            return r;
        }

        /**
         * harmonic mean
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         */
        template<RealType dType>
        auto harmmean(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<real_max> {
            auto dim_idx = dim - 1;
            if (m.isEmpty())
                return {};
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0) {
                real_max sum = 0.;
                for (auto &i: m)
                    sum += i;
                return {m.getSize() / sum};
            }
            auto shape = m.getShape();
            auto step_size = matsar::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<real_max> r(r_shape, 1, m.getData());

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
                    item = (real_max) step_num / item;
                }
                iter_r += step_size;
            }
            return r;
        }

        template<Calculable dType>
        auto trace(const Matrix<dType> &m) -> dType {
            if (!m.isSquare() || m.getDims() > 2)
                throw_value_error("trace requires the matrix to be square with 2 dimensions");
            dType r = 0;
            for (msize_type i = 0; i < m.getShape()[0]; ++i)
                r += m(i, i);
            return r;
        }
    }

///LU decomposition
    namespace Math {
        auto LAPACK_LU(lapack_int m, lapack_int n, float64 *data, lapack_int lda, lapack_int *ipiv) -> lapack_int {
            return LAPACKE_dgetrf2(CblasColMajor, m, n, data, lda, ipiv);
        }

        auto LAPACK_LU(lapack_int m, lapack_int n, complex64 *data, lapack_int lda, lapack_int *ipiv) -> lapack_int {
            return LAPACKE_cgetrf2(CblasColMajor, m, n, data, lda, ipiv);
        }

        auto LAPACK_LU(lapack_int m, lapack_int n, complex128 *data, lapack_int lda, lapack_int *ipiv) -> lapack_int {
            return LAPACKE_zgetrf2(CblasColMajor, m, n, data, lda, ipiv);
        }

        template<Calculable TargetType, Calculable dType>
        auto lu(const Matrix<dType> &M) -> tuple<Matrix<TargetType>, Matrix<TargetType>> {
            Matrix<TargetType> fm(M);
            fm.expandShape(2);
            if (!fm.is2D())
                throw_value_error("lu requires a 2D matrix or a matrix that can be expanded to 2D");
            lapack_int m = fm.getShape()[0], n = fm.getShape()[1];
            lapack_int min_m_n = m > n ? std::max(m, n) : std::min(m, n);
            unique_ptr<lapack_int[]> IPIV(new lapack_int[min_m_n]);
            memset(IPIV.get(), -1, min_m_n);

            LAPACK_LU(m, n, fm.getData(), m, IPIV.get());

            unique_ptr<lapack_int[]> IDX(new lapack_int[min_m_n]);
            for (int i = 0; i < min_m_n; ++i)
                IDX[i] = i;
            for (int i = 0; i < min_m_n; ++i) {
                lapack_int swap_idx = IPIV[i] - 1;
                if (swap_idx != i && swap_idx >= 0 && swap_idx < min_m_n)
                    std::swap(IDX[i], IDX[swap_idx]);
            }
            Matrix<int64> P({min_m_n, min_m_n}, 0., M.getData());
            for (int i = 0; i < min_m_n; ++i)
                P(IDX[i], i) = 1;//P.T()!!!

//            cout << fm.toRowColString("fm") << endl;
//            cout << P.toString("p") << endl;
            // P is the row-permutation matrix; applying it to L is a MATRIX product,
            // not element-wise -- use mmul/operator% (operator* is Hadamard). Normalize
            // to Matrix<TargetType> since mmul promotes via its (x + 0.) default type.
            if (m == n) {
                Matrix<TargetType> L = mmul(P, tril(fm, -1) + eye<TargetType>(m, n));
                return {L, triu(fm)};
            } else if (m > n) {
                Matrix<TargetType> L = mmul(P, tril(fm, -1) + eye<TargetType>(m, n));
                auto U = triu(fm)({Range<msize_type>(n), Range<msize_type>(n)});
                return {L, U};
            } else {
                Matrix<TargetType> PL = mmul(P, tril(fm, -1) + eye<TargetType>(m, n));
                auto L = PL({Range<msize_type>(m), Range<msize_type>(m)});
                return {L, triu(fm)};
            }
        }

        template<RealType RT>
        auto lu(const Matrix<RT> &M) -> tuple<Matrix<float64>, Matrix<float64>> {
            return lu<float64, float64>(Matrix<float64>(M));
        }

        auto lu(const Matrix<complex64> &M) -> tuple<Matrix<complex64>, Matrix<complex64>> {
            return lu<complex64, complex64>(M);
        }

        auto lu(const Matrix<complex128> &M) -> tuple<Matrix<complex128>, Matrix<complex128>> {
            return lu<complex128, complex128>(M);
        }

        template<Calculable TargetType, Calculable dType>
        auto lup(const Matrix<dType> &M) -> tuple<Matrix<TargetType>, Matrix<TargetType>, Matrix<int64>> {
            Matrix<TargetType> fm(M);
            fm.expandShape(2);
            if (!fm.is2D())
                throw_value_error("lup requires a 2D matrix or a matrix that can be expanded to 2D");
            lapack_int m = fm.getShape()[0], n = fm.getShape()[1];
            lapack_int min_m_n = m > n ? std::max(m, n) : std::min(m, n);
            unique_ptr<lapack_int[]> IPIV(new lapack_int[min_m_n]);
            memset(IPIV.get(), -1, min_m_n);

            LAPACK_LU(m, n, fm.getData(), m, IPIV.get());
//            LAPACKE_dgetrf2(CblasColMajor, m, n, fm.getData(), m, IPIV.get());

            unique_ptr<lapack_int[]> IDX(new lapack_int[min_m_n]);
            for (int i = 0; i < min_m_n; ++i)
                IDX[i] = i;
            for (int i = 0; i < min_m_n; ++i) {
                lapack_int swap_idx = IPIV[i] - 1;
                if (swap_idx != i && swap_idx >= 0 && swap_idx < min_m_n)
                    std::swap(IDX[i], IDX[swap_idx]);
            }
            Matrix<int64> P({min_m_n, min_m_n}, 0., M.getData());
            for (int i = 0; i < min_m_n; ++i)
                P(i, IDX[i]) = 1;//P!!!
//            cout << fm.toRowColString("fm") << endl;
//            return {(tril(fm, -1) + eye<TargetType>(m, n)), triu(fm), P};
            if (m == n)
                return {(tril(fm, -1) + eye<TargetType>(m, n)), triu(fm), P};
            else if (m > n) {
                auto U = (triu(fm))({Range<msize_type>(n), Range<msize_type>(n)});
                return {tril(fm, -1) + eye<TargetType>(m, n), U, P};
            } else {
                auto L = (tril(fm, -1) + eye<TargetType>(m, n))({Range<msize_type>(m), Range<msize_type>(m)});
                return {L, triu(fm), P};
            }
        }

        template<RealType RT>
        auto lup(const Matrix<RT> &M) -> tuple<Matrix<float64>, Matrix<float64>, Matrix<int64>> {
            return lup<float64, RT>(M);
        }

        auto lup(const Matrix<complex64> &M) -> tuple<Matrix<complex64>, Matrix<complex64>, Matrix<int64>> {
            return lup<complex64, complex64>(M);
        }

        auto lup(const Matrix<complex128> &M) -> tuple<Matrix<complex128>, Matrix<complex128>, Matrix<int64>> {
            return lup<complex128, complex128>(M);
        }
    }

    namespace Math {
        template<RealType RT>
        auto det(const Matrix<RT> &m) -> Matrix<real_max> {
            auto lup_res = lup(m);
            auto &L = std::get<0>(lup_res);
            auto &U = std::get<1>(lup_res);
            return L.det() * prod(diag(U));
        }

        template<ComplexType CT>
        auto det(const Matrix<CT> &m) -> Matrix<complex_max> {
            auto lup_res = lup(m);
            auto &L = std::get<0>(lup_res);
            auto &U = std::get<1>(lup_res);
            return L.det() * prod(diag(U));
        }
    }

    namespace Math {
        // element type the LAPACK routines compute in: real_max for real inputs
        // (integers included), complex_max for complex inputs.
        template<typename T>
        using la_promote_t = std::conditional_t<matsar::complex_type<T>::value, complex_max, real_max>;

        template<Calculable dType>
        auto inv(const Matrix<dType> &m) -> Matrix<la_promote_t<dType>> {
            using Out = la_promote_t<dType>;
            if (!m.isSquare() || m.getDims() != 2)
                throw_value_error("matrix inversion requires a 2D square matrix");

            auto n = m.getShape()[0];
            Matrix<Out> A(m), X({n, n}, 0., m.getData());
            lapack_int iter, info;
            unique_ptr<lapack_int[]> IPIV(new lapack_int[n]);

            if constexpr (matsar::complex_type<dType>::value)      // <- the ONLY real/complex difference
                info = LAPACKE_zcgesv(CblasColMajor, n, n, A.getData(), n, IPIV.get(),
                                      eye<Out>(n).getData(), n, X.getData(), n, &iter);
            else
                info = LAPACKE_dsgesv(CblasColMajor, n, n, A.getData(), n, IPIV.get(),
                                      eye<Out>(n).getData(), n, X.getData(), n, &iter);

            if (info < 0)
                throw_value_error("inv argument error " + to_string(info));
            if (info > 0) {
                stringstream ss;
                ss << "matrix rank is " << rank(m) << " shape is " << n << " has no inverse";
                m.debug();
                throw_warning(ss.str());
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
        template<typename dType>
        auto eig(const Matrix<dType> &A) -> vector<Matrix<float64>> {
            Matrix<float64> fA(A);
            fA.expandShape(2);
            if (!fA.is2D())
                throw_value_error("eig requires a 2D matrix or a matrix that can be expanded to 2D");
            lapack_int m = fA.getShape()[0], n = fA.getShape()[1];
            mshape_type S_shape = {min(m, n)};
            Matrix<float64> S(S_shape, 0., A.getData());
            unique_ptr<float64> size(new float64);
            unique_ptr<float64[]> wr(new float64[m]);
            unique_ptr<float64[]> wi(new float64[m]);
            unique_ptr<float64[]> vl(new float64[m * n]);
            unique_ptr<float64[]> vr(new float64[m * n]);
            int lda = (int) m;
            int ldvl = (int) m;
            int ldvr = (int) n;
            auto info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'V', 'V', m, fA.getData(), lda, wr.get(), wi.get(),
                                      vl.get(),
                                      ldvl, vr.get(), ldvr);
            if (info != 0) {
                A.debug();
                throw_warning("eig works badly error code:" + to_string(info));
            }
            vector<Matrix<float64>> V_D_W(3);
            vector<float64> eig_v;
            for (int i = 0; i < m; i++) {
                eig_v.push_back(wr.get()[i]);
            }

            vector<float64> l_eigenvector;
            vector<float64> r_eigenvector;

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    l_eigenvector.push_back(vl.get()[n * j + i]);
                    r_eigenvector.push_back(vr.get()[n * j + i]);
                }
            }

            Matrix<float64> V(l_eigenvector);
            Matrix<float64> W(r_eigenvector);
            V = V.reshape({m, n});
            W = W.reshape({m, n});
            V_D_W[0] = V;
            V_D_W[1] = diag(Matrix<float64>(eig_v));
            V_D_W[2] = W;
            return V_D_W;
        }

        /**
         * @author wzp
         * @function condeig condition numbers related to the eigenvalues
         * @return [V,D,s] = condeig(A)
         * s = a vector of condition numbers for the eigenvalues of A. These condition numbers are the reciprocals of the cosines of the angles between the left and right eigenvectors.
         */
        template<typename dType>
        auto condeig(const Matrix<dType> &A) -> vector<Matrix<float64>> {
            vector<Matrix<dType>> V_D_s(3);
            if (!A.is2D() || !A.isSquare()) {
                throw_value_error("input matrix must be a square matrix!");
                return V_D_s;
            }
            auto V_D_W = eig(A);
            auto V = V_D_W[0];
            auto D = V_D_W[1];
            auto W = V_D_W[2];
            auto shape = D.getShape();
            vector<dType> s(shape[0]);
            vector<dType> a(shape[1]);
            vector<dType> b(shape[1]);

            for (int j = 0; j < shape[1]; j++) {
                float64 mul = 0;
                float64 norm_a = 0;
                float64 norm_b = 0;
                for (int i = 0; i < shape[0]; i++) {
                    mul += V(j, i) * W(j, i);
                    norm_a += std::pow(V(j, i), 2);
                    norm_b += std::pow(W(j, i), 2);
                }
                s[j] = std::sqrt(norm_a) * std::sqrt(norm_b) / mul;
            }
            V_D_s[0] = V;
            V_D_s[1] = D;
            V_D_s[2] = Matrix<dType>(s);
            return V_D_s;
        }

        //pinv function
        template<typename dType>
        auto pinv(const Matrix<dType> &matrix) -> Matrix<dType> {
            Matrix<float64> transposed = matrix.T();
            int rows = matrix.getShape()[0];  //rows
            int cols = matrix.getShape()[1];  //cols
            if (rows > cols) {
                Matrix<float64> product = mmul(transposed, matrix);
                Matrix<float64> inverse = inv(product);
                Matrix<float64> res = mmul(inverse, transposed);
                //res = res.reshape({ n*origRows,n*origCols });
                return res;
            } else {
                Matrix<float64> product = mmul(matrix, transposed);
                Matrix<float64> inverse = inv(product);
                Matrix<float64> res = mmul(transposed, inverse);
                //res = res.reshape({ n*origRows,n*origCols });
                return res;
            }
        }

        template<typename dType>
        Matrix<dType> schur(const Matrix<dType> &A, const std::string &flag) {
            Matrix<dType> inputA(A);
            inputA.expandShape(2);

            if (!inputA.is2D()) {
                throw_value_error("schur requires a 2D matrix or a matrix that can be expanded to 2D");
            }

            int m = inputA.getShape()[0];
            int n = inputA.getShape()[1];

            // Ensure A is a square matrix
            if (m != n) {
                throw_value_error("schur requires a square matrix");
            }

            lapack_int lda = (int) m;
            lapack_int sdim;
            unique_ptr<dType[]> wr(new dType[m]);
            unique_ptr<dType[]> wi(new dType[m]);
            unique_ptr<dType[]> vs(new dType[m * n]);

            int ldvs = (int) m;
            char sort = 'N';
            LAPACK_D_SELECT2 select = nullptr;
            auto info = LAPACKE_dgees(LAPACK_COL_MAJOR, 'V', sort, select, n, inputA.getData(), lda, &sdim, wr.get(),
                                      wi.get(), vs.get(), ldvs);

            if (info != 0) {
                A.debug();
                throw_warning("schur works badly, error code:" + to_string(info));
            }

            Matrix<dType> T = inputA;

            if (flag == "complex") {
                return T;
            } else if (flag == "real") {
                // Convert T to real Schur form
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < i; j++) {
                        T(i, j) = 0;
                    }
                }
            } else {
                throw_value_error("Invalid flag value. Expected 'complex' or 'real'.");
            }
            //T still needs transposing
            return T;
        }

        /******schur by wwl******/
        template<typename dType>
        vector<Matrix<dType>> schur(const Matrix<dType> &A) {
            vector<Matrix<dType>> U_T(2); //store the result

            Matrix<dType> inputA(A);
            inputA.expandShape(2);

            if (!inputA.is2D()) {
                throw_value_error("schur requires a 2D matrix or a matrix that can be expanded to 2D");
            }

            int m = inputA.getShape()[0];
            int n = inputA.getShape()[1];

            // Ensure A is a square matrix
            if (m != n) {
                throw_value_error("schur requires a square matrix");
            }

            lapack_int lda = (int) m;
            lapack_int sdim;
            unique_ptr<dType[]> wr(new dType[m]);
            unique_ptr<dType[]> wi(new dType[m]);
            unique_ptr<dType[]> vs(new dType[m * n]);

            int ldvs = (int) m;
            char sort = 'N';
            LAPACK_D_SELECT2 select = nullptr;
            auto info = LAPACKE_dgees(LAPACK_COL_MAJOR, 'V', sort, select, n, inputA.getData(), lda, &sdim, wr.get(),
                                      wi.get(), vs.get(), ldvs);

            if (info != 0) {
                A.debug();
                throw_warning("schur works badly, error code:" + to_string(info));
            }

            Matrix<dType> T = inputA;
            vector<dType> temp_U;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    temp_U.push_back(vs.get()[(m - 1) * n + i - j * n]);
                }
            }
            Matrix<dType> U(temp_U);
            U = U.reshape({m, n});

            // T,U still need transposing
            U_T[0] = T;
            U_T[1] = U;
            return U_T;
        }

    }

    namespace Math {
        template<typename dType>
        auto polyder(const vector<dType> &p) -> vector<dType> {
            if (p.size() <= 1) return {0};
            vector<dType> r(p);
            r.pop_back();
            auto n = r.size();
            for (auto &i: r) {
                if (i != 0.)
                    i *= n;
                n--;
            }
            return r;
        }

        /**
         * @author zzx
         */
        template<typename dType>
        auto polyder(const vector<dType> &a, const vector<dType> &b) -> vector<dType> {
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

        /**
         * @author zzx
         */
        template<typename dType>
        auto polyvalm(const vector<dType> &p, Matrix<dType> &X) -> Matrix<dType> {
            if (!X.isSquare()) {
                throw_value_error("polyvalm:NonSquareMatrix");
            }
            auto m = X.getShape()[0];
            Matrix<dType> Y(X.getShape(), 0.);
            for (int i = 0; i < p.size(); ++i) {
                Y = mmul(X, Y) + diag(p[i] * ones(mshape_type{m}));
            }
            return Y;
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
         */
        template<typename dType>
        auto ppval(const vector<dType> &breaks, Matrix<dType> coefs, const vector<dType> &xq) -> vector<dType> {
            coefs.expandShape(2);
            auto coefs_shape = coefs.getShape();
            if (breaks.size() - 1 != coefs_shape[1])
                throw_value_error("breaks size must match coefs size");
            if (breaks.size() - 1 == 0)
                throw_value_error("breaks size must bigger than one");
            vector<vector<dType>> coefs_v(coefs_shape[1], vector<dType>(coefs_shape[0], 0));
            for (int64 i = 0; i < coefs_shape[0]; ++i) {
                for (int64 j = 0; j < coefs_shape[1]; ++j) {
                    coefs_v[j][i] = coefs(i, j);
                }
            }

            vector<dType> r(xq.size(), 0.);
            const dType left_bound = *breaks.begin();
            const dType right_bound = *(breaks.end() - 1);
            for (typename vector<dType>::size_type xi = 0; xi < xq.size(); ++xi) {
                auto x = xq[xi];
                if (x <= left_bound) {
                    r[xi] = polyeval(*coefs_v.begin(), x - *breaks.begin());
                } else if (x >= right_bound) {
                    r[xi] = polyeval(*(coefs_v.end() - 1), x - *(breaks.end() - 2));
                } else {
                    for (int64 i = 0; i < coefs_v.size(); ++i) {
                        if (x >= breaks[i] && x <= breaks[i + 1])
                            r[xi] = polyeval(coefs_v[i], x - breaks[i]);
                    }
                }
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
        }

        /**
         * @author zzx
         * MATLAB-style exponential integral: expint(x) = E1(x) = -Ei(-x),
         * computed via the native Internal::expint_ei above.
         */
        template<typename dType, typename TargetType = real_max>
        auto Expint(const Matrix<dType> x) -> Matrix<TargetType> {
            Matrix<TargetType> r(x);
            for (auto &i: r) {
                i = -Internal::expint_ei(-i);
            }
            return r;
        }

        /**
         * @author zzx
         */
        template<typename dType>
        auto Expint(const Matrix<complex<dType>> x) -> Matrix<complex<dType>> {
            Matrix<complex<dType>> r(x);
            for (auto &i: r) {
                i = -Internal::expint_ei(-i.real()) - pi * I(1.);
            }
            return r;
        }

        auto spdiags(const Matrix<double> &A) -> tuple<Matrix<double>, Matrix<double>> {
            Matrix<double> temp(A);
            int m = temp.getShape()[1], n = temp.getShape()[0];
            if (m == 0 || n == 0) {
                return {};
            }
            vector<double> idx;
            int num = min(m, n);
            vector<vector<double>> res;
            int k = -m + 1;
            //m >=n below the diagonal counts from one; m<n, the part below the diagonal must first be filled with 0
            if (m >= n) {
                for (int row = m - 1; row >= 0; row--) {
                    bool flag = false;
                    vector<double> temp_v(num, 0.);
                    for (int i = row, j = 0, index = 0; i < m && j < n; i++, j++) {
                        if (temp(j, i) != 0.) {
                            flag = true;
                        }
                        temp_v[index++] = temp(j, i);
                    }
                    //only added when there is a non-0
                    if (flag) {
                        idx.push_back(k);
                        res.push_back(temp_v);
                    }
                    k++;
                }
                for (int col = 1; col < n; col++) {
                    vector<double> temp_v(num, 0.);
                    bool flag = false;
                    //k>0 is the upper-triangular part
                    for (int i = 0, j = col, index = k; i < m && j < n; i++, j++) {
                        if (temp(j, i) != 0.) {
                            flag = true;
                        }
                        temp_v[index++] = temp(j, i);
                    }
                    if (flag) {
                        res.push_back(temp_v);
                        idx.push_back(k);
                    }
                    k++;
                }
            } else {
                for (int row = m - 1; row >= 0; row--) {
                    bool flag = false;
                    vector<double> temp_v(num, 0.);
                    for (int i = row, j = 0, index = -k; i < m && j < n; i++, j++) {
                        if (temp(j, i) != 0.) {
                            flag = true;
                        }
                        temp_v[index++] = temp(j, i);
                    }
                    //only added when there is a non-0
                    if (flag) {
                        idx.push_back(k);
                        res.push_back(temp_v);
                    }
                    k++;
                }
                for (int col = 1; col < n; col++) {
                    vector<double> temp_v(num, 0.);
                    bool flag = false;
                    //k>0 is the upper-triangular part
                    for (int i = 0, j = col, index = 0; i < m && j < n; i++, j++) {
                        if (temp(j, i) != 0.) {
                            flag = true;
                        }
                        temp_v[index++] = temp(j, i);
                    }
                    if (flag) {
                        res.push_back(temp_v);
                        idx.push_back(k);
                    }
                    k++;
                }
            }
            Matrix<double> res_m(res);
            return std::make_tuple(res_m.T(), idx);
        }

        auto spdiags(const Matrix<double> A, const Matrix<double> d) -> Matrix<double> {
            if (A.getSize() < 1 || d.getSize() < 1) {
                return {};
            }
            Matrix<double> temp(A.T());
            int m = temp.getShape()[0], n = temp.getShape()[1];
            int len = min(m, n);
            vector<vector<double>> res_v;
            for (auto iter_d = d.begin(); iter_d != d.end(); iter_d++) {
                int k = (int) (*iter_d);
                vector<double> temp_v(len, 0.);
                if (k <= 0 && k > -m) {
                    for (int i = -k, j = 0, idx = 0; i < m && j < n; i++, j++) {
                        temp_v[idx++] = temp(i, j);
                    }
                } else if (k > 0 && k < n) {
                    for (int i = 0, j = k, idx = k; i < m && j < n; i++, j++) {
                        temp_v[idx++] = temp(i, j);
                    }
                }
                res_v.push_back(temp_v);
            }
            Matrix<double> res(res_v);
            return res.T();
        }

        template<typename T, typename FT>
        auto funm(const Matrix<T> &A, FT matrix_fun, const string &fun_name) -> Matrix<T> {
            vector<Matrix<T>> V_D_W = eig(A);
            auto inter_m = diag(V_D_W[1]);
            matrix_fun(inter_m, fun_name);
            return mmul(mmul(inv(V_D_W[0]), diag(inter_m)), (V_D_W[0]));
        }
    }
}

#endif //MATSAR_LINEARALGEBRA_HPP
