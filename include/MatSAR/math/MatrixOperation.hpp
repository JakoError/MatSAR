// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2022/7/10.
//
///Extension of the matrix sign function
#ifndef MATSAR_MATRIXOPERATION_HPP
#define MATSAR_MATRIXOPERATION_HPP

#include <numeric>
#include <algorithm>
#include <unordered_map>

// NOTE: MatrixOperation calls no CBLAS/LAPACK routines directly -- matrix multiply
// (mmul / operator%) lives in Matrix.hpp, which pulls in <cblas.h> itself, guarded
// by MATSAR_WITH_BLAS. Including <cblas.h> unconditionally here broke no-BLAS builds
// and hard-required the header even when the core was built without BLAS.
#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

#ifndef MATSAR_TRIGONOMETRY_HPP

#include "Trigonometry.hpp"

#endif

// nnz() delegates to sum(); pull that in directly (MathematicalFunction) rather than
// the heavyweight, LAPACK-dependent LinearAlgebra (which MatrixOperation does not use --
// its only det/inv references are commented out below). Keeps MatrixOperation, and its
// downstream includers like Probability, buildable without BLAS.
#ifndef MATSAR_MATHEMATICALFUNCTION_HPP

#include "MathematicalFunction.hpp"

#endif

namespace matsar {
    ///complex
    namespace Math {
        template<typename dType>
        inline auto real(const Matrix<dType> &m) -> decltype(m.real()) {
            return m.real();
        }

        template<typename dType>
        inline auto image(const Matrix<dType> &m) -> decltype(m.image()) {
            return m.image();
        }

        template<typename dType>
        inline auto conj(const Matrix<dType> &m) -> decltype(m.conj()) {
            return m.conj();
        }
    }

    namespace Math {
        template<typename dType, typename TargetType = decltype(matsar::ceil(std::declval<dType>()))>
        static auto ceil(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::ceil(x); });
        }

        template<typename dType, typename TargetType = decltype(matsar::floor(std::declval<dType>()))>
        static auto floor(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::floor(x); });
        }

        /**
         * x rounded to n decimal digits, half away from zero (MATLAB round(x, n); n < 0 rounds
         * to tens, hundreds, ...). The exact stored value of x is rounded, as Python's round()
         * does: 2.675 is stored as 2.67499999999999982..., so round(2.675, 2) == 2.67.
         * x is scaled by an exact power of ten p and the integer result divided back by p (the
         * old multiplication by 0.1^n was inexact: round(0.3, 1) gave 0.30000000000000004). The
         * scaled value is itself rounded, but that can only put it on the wrong side of a tie
         * when it lands exactly on one (k + 1/2); the exact rounding error (from fma) then
         * tells which side the true value is on.
         */
        template<typename F>
        auto round_digits(F x, int n) -> F {
            if (n == 0 || !std::isfinite(x))
                return static_cast<F>(matsar::round(x));
            const F p = static_cast<F>(std::pow(10.0, std::abs(n)));   // exact up to 10^22
            if (!std::isfinite(p))
                return n > 0 ? x : x * F(0);
            F scaled, err;   // err: the sign of (exact x * 10^n) - scaled
            if (n > 0) {
                scaled = x * p;
                // |x| * 10^n >= 2^digits: x has no digit at the 10^-n place left to drop
                if (!(std::abs(scaled) < std::ldexp(F(1), std::numeric_limits<F>::digits)))
                    return x;
                err = std::fma(x, p, -scaled);    // x * p - scaled, exactly
            } else {
                scaled = x / p;
                err = std::fma(-scaled, p, x);    // x - scaled * p, exactly (sign of x/p - scaled)
            }
            const F t = std::trunc(scaled);
            F r = static_cast<F>(matsar::round(scaled));
            if (std::abs(scaled - t) == F(0.5) && err != 0 && (err < 0) != (scaled < 0))
                r = t;   // the exact value lies just inside the tie: round toward zero
            return n > 0 ? r / p : r * p;
        }

        template<typename F>
        auto round_digits(const std::complex<F> &x, int n) -> std::complex<F> {
            return {round_digits(x.real(), n), round_digits(x.imag(), n)};
        }

        /**
         * round(m), or round(m, n) to n decimal digits. The result is floating point (an integer
         * input is converted first: it used to be truncated to an integer before rounding).
         */
        template<typename dType, typename TargetType = decltype(matsar::round(std::declval<dType>()))>
        static auto round(const Matrix<dType> &m, int n = 0) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(
                    m, [n](const dType &x) { return round_digits(static_cast<TargetType>(x), n); });
        }

        template<typename dType, typename TargetType = decltype(matsar::fix(std::declval<dType>()))>
        static auto fix(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::fix(x); });
        }

        template<IntType dType CheckInt(dType)>
        static auto isPrime(const Matrix<dType> &m) -> Matrix<bool> {
            return apply_function<dType, bool>(m, [](const dType &x) { return matsar::isPrime(x); });
        }

        template<IntType IT1, IntType IT2 CheckInt(IT1) CheckInt(IT2)>
        static auto numden(const Matrix<IT1> &m1, const Matrix<IT2> &m2) -> vector<Matrix<int64>> {
            Matrix<int64> r1(m1);
            Matrix<int64> r2(m2);
            Matrix<int64>::broadcast(r1, r2);
            for (auto iter1 = r1.begin(), iter2 = r2.begin(); iter1 != r1.end(); ++iter1, ++iter2) {
                auto values = matsar::numden(*iter1, *iter2);
                *iter1 = values[0];
                *iter2 = values[1];
            }
            return {r1, r2};
        }
    }

    ///special function
    namespace Math {
        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Gcd(leftT *left, rightT *right, msize_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = matsar::gcd(*left++, *right++);
        }

        template<typename IT1, typename IT2, typename TargetType = decltype(gcd(declval<IT1>(), declval<IT2>()))>
        auto gcd(const Matrix<IT1> &m1, const Matrix<IT2> &m2) -> Matrix<TargetType> {
            if (m1.getShape() == m2.getShape()) {
                Matrix<TargetType> r(m1.getShape(), 0., m1.getData());
                Gcd(m1.getData(), m2.getData(), r.getSize(), r.getData());
                return r;
            } else {
                auto b_m1 = m1;
                auto b_m2 = m2;
                Matrix<TargetType>::broadcast(b_m1, b_m2);
                Matrix<TargetType> r(b_m1.getShape(), 0., m1.getData());
                Gcd(b_m1.getData(), b_m2.getData(), r.getSize(), r.getData());
                return r;
            }
        }

        ///FIXME Optimizable factorial algorithm using array to record data
        template<typename dType, typename TargetType = float_max>
        auto factorial(const Matrix<dType> &m) -> Matrix<TargetType> {
            Matrix<TargetType> r(m);
            for (auto &i: r)
                i = matsar::factorial<TargetType>(i);
            return r;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Nchoosek(leftT *left, rightT *right, msize_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = matsar::nchoosek<leftT, rightT, DstT>(*left++, *right++);
        }

        template<typename IT1, typename IT2, typename TargetType = float_max>
        auto nchoosek(const Matrix<IT1> &m1, const Matrix<IT2> &m2) -> Matrix<TargetType> {
            if (m1.getShape() == m2.getShape()) {
                Matrix<TargetType> r(m1.getShape(), 0., m1.getData());
                Nchoosek(m1.getData(), m2.getData(), r.getSize(), r.getData());
                return r;
            } else {
                auto b_m1 = m1;
                auto b_m2 = m2;
                Matrix<TargetType>::broadcast(b_m1, b_m2);
                Matrix<TargetType> r(b_m1.getShape(), 0., m1.getData());
                Nchoosek(b_m1.getData(), b_m2.getData(), r.getSize(), r.getData());
                return r;
            }
        }
    }

    ///logx exp
    namespace Math {
        template<typename dType, typename TargetType = decltype(matsar::log(std::declval<dType>()))>
        auto log(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::log(x); });
        }

        template<typename dType, typename TargetType = decltype(matsar::log(std::declval<dType>()))>
        auto ln(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::log(x); });
        }

        template<typename dType, typename TargetType = decltype(matsar::log2(std::declval<dType>()))>
        auto log2(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::log2(x); });
        }

        template<typename dType, typename TargetType = decltype(matsar::log10(std::declval<dType>()))>
        auto log10(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::log10(x); });
        }

        template<typename dType, typename TargetType = decltype(matsar::exp(std::declval<dType>()))>
        auto exp(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::exp(x); });
        }

        template<typename dType, typename TargetType = decltype(matsar::exp2(std::declval<dType>()))>
        auto exp2(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::exp2(x); });
        }
    }
    ///gamma
    namespace Math {
        template<typename dType, typename TargetType = decltype(matsar::gamma(std::declval<dType>()))>
        auto gamma(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::gamma(x); });
        }

        template<typename dType, typename TargetType = decltype(matsar::gammaln(std::declval<dType>()))>
        auto gammaln(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::gammaln(x); });
        }
    }

    ///mpower sqrt
    namespace Math {
        template<typename dType, typename TargetType = decltype(matsar::sqrt(std::declval<dType>()))>
        auto sqrt(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, [](const dType &x) { return matsar::sqrt(x); });
        }

    }

    ///matrix functions: expm logm mpower (functions of the whole matrix, as in MATLAB; the
    ///element-wise forms are exp, log and the ^ operator). Plain dense kernels, no BLAS.
    namespace Math {
        namespace MatFun {
            /// an n x n matrix in column-major order: A[i + j * n] is element (i, j)
            template<typename T>
            using Dense = std::vector<T>;

            template<typename T>
            auto identity(msize_type n) -> Dense<T> {
                Dense<T> I(n * n, T(0));
                for (msize_type i = 0; i < n; ++i)
                    I[i + i * n] = T(1);
                return I;
            }

            template<typename T>
            auto mul(msize_type n, const Dense<T> &A, const Dense<T> &B) -> Dense<T> {
                Dense<T> C(n * n, T(0));
                for (msize_type j = 0; j < n; ++j)
                    for (msize_type k = 0; k < n; ++k) {
                        const T b = B[k + j * n];
                        for (msize_type i = 0; i < n; ++i)
                            C[i + j * n] += A[i + k * n] * b;
                    }
                return C;
            }

            /// a * A + b * B
            template<typename T>
            auto axpby(const T &a, const Dense<T> &A, const T &b, const Dense<T> &B) -> Dense<T> {
                Dense<T> C(A.size());
                for (size_t i = 0; i < A.size(); ++i)
                    C[i] = a * A[i] + b * B[i];
                return C;
            }

            /// 1-norm (largest column sum of |a_ij|); NaN if any element is NaN
            template<typename T>
            auto norm1(msize_type n, const Dense<T> &A) -> float64 {
                float64 best = 0;
                for (msize_type j = 0; j < n; ++j) {
                    float64 col = 0;
                    for (msize_type i = 0; i < n; ++i)
                        col += static_cast<float64>(std::abs(A[i + j * n]));
                    if (std::isnan(col))
                        return col;
                    best = std::max(best, col);
                }
                return best;
            }

            /// X = A^-1 B by Gaussian elimination with partial pivoting; false if A is singular
            template<typename T>
            auto solve(msize_type n, Dense<T> A, Dense<T> B, Dense<T> &X) -> bool {
                for (msize_type k = 0; k < n; ++k) {
                    msize_type piv = k;
                    float64 big = static_cast<float64>(std::abs(A[k + k * n]));
                    for (msize_type i = k + 1; i < n; ++i) {
                        auto v = static_cast<float64>(std::abs(A[i + k * n]));
                        if (v > big) {
                            big = v;
                            piv = i;
                        }
                    }
                    if (big == 0)
                        return false;
                    if (piv != k)
                        for (msize_type j = 0; j < n; ++j) {
                            std::swap(A[k + j * n], A[piv + j * n]);
                            std::swap(B[k + j * n], B[piv + j * n]);
                        }
                    for (msize_type i = k + 1; i < n; ++i) {
                        const T f = A[i + k * n] / A[k + k * n];
                        for (msize_type j = k + 1; j < n; ++j)
                            A[i + j * n] -= f * A[k + j * n];
                        for (msize_type j = 0; j < n; ++j)
                            B[i + j * n] -= f * B[k + j * n];
                    }
                }
                for (msize_type j = 0; j < n; ++j)
                    for (msize_type i = n; i-- > 0;) {
                        T acc = B[i + j * n];
                        for (msize_type k = i + 1; k < n; ++k)
                            acc -= A[i + k * n] * B[k + j * n];
                        B[i + j * n] = acc / A[i + i * n];
                    }
                X = std::move(B);
                return true;
            }

            /**
             * exp(A): scaling and squaring with a [m/m] Pade approximant, m in {3, 5, 7, 9, 13}
             * chosen from ||A||_1 (N. J. Higham, "The scaling and squaring method for the matrix
             * exponential revisited", SIAM J. Matrix Anal. Appl. 26(4), 2005).
             */
            template<typename T>
            auto expm(msize_type n, Dense<T> A) -> Dense<T> {
                const float64 norm = norm1(n, A);
                if (!std::isfinite(norm))
                    return Dense<T>(n * n, T(std::numeric_limits<float64>::quiet_NaN()));
                static const float64 b3[] = {120, 60, 12, 1};
                static const float64 b5[] = {30240, 15120, 3360, 420, 30, 1};
                static const float64 b7[] = {17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1};
                static const float64 b9[] = {17643225600., 8821612800., 2075673600., 302702400., 30270240.,
                                             2162160., 110880., 3960., 90., 1.};
                static const float64 b13[] = {64764752532480000., 32382376266240000., 7771770303897600.,
                                              1187353796428800., 129060195264000., 10559470521600.,
                                              670442572800., 33522128640., 1323241920., 40840800., 960960.,
                                              16380., 182., 1.};
                static const float64 theta[] = {1.495585217958292e-2, 2.539398330063230e-1,
                                                9.504178996162932e-1, 2.097847961257068e0};
                static const float64 *low_b[] = {b3, b5, b7, b9};
                const float64 theta13 = 5.371920351148152e0;
                const auto I = identity<T>(n);
                Dense<T> U, V;
                int squarings = 0;
                int deg = -1;
                for (int d = 0; d < 4; ++d)
                    if (norm <= theta[d]) {
                        deg = d;
                        break;
                    }
                if (deg >= 0) {
                    // U = A * sum b[2k+1] A^2k, V = sum b[2k] A^2k, k = 0 .. m/2
                    const float64 *b = low_b[deg];
                    const int m = 2 * deg + 3;
                    const auto A2 = mul(n, A, A);
                    Dense<T> P = I, Uo(n * n, T(0));
                    V.assign(n * n, T(0));
                    for (int k = 0; 2 * k <= m; ++k) {
                        if (k > 0)
                            P = mul(n, P, A2);
                        Uo = axpby(T(1), Uo, T(b[2 * k + 1]), P);
                        V = axpby(T(1), V, T(b[2 * k]), P);
                    }
                    U = mul(n, A, Uo);
                } else {
                    squarings = std::max(0, static_cast<int>(std::ceil(std::log2(norm / theta13))));
                    const T scale = T(std::ldexp(1.0, -squarings));
                    for (auto &a: A)
                        a *= scale;
                    const auto A2 = mul(n, A, A), A4 = mul(n, A2, A2), A6 = mul(n, A2, A4);
                    const auto *b = b13;
                    auto inner_u = axpby(T(b[13]), A6, T(b[11]), A4);
                    inner_u = axpby(T(1), inner_u, T(b[9]), A2);
                    auto Uo = mul(n, A6, inner_u);
                    Uo = axpby(T(1), Uo, T(b[7]), A6);
                    Uo = axpby(T(1), Uo, T(b[5]), A4);
                    Uo = axpby(T(1), Uo, T(b[3]), A2);
                    Uo = axpby(T(1), Uo, T(b[1]), I);
                    U = mul(n, A, Uo);
                    auto inner_v = axpby(T(b[12]), A6, T(b[10]), A4);
                    inner_v = axpby(T(1), inner_v, T(b[8]), A2);
                    V = mul(n, A6, inner_v);
                    V = axpby(T(1), V, T(b[6]), A6);
                    V = axpby(T(1), V, T(b[4]), A4);
                    V = axpby(T(1), V, T(b[2]), A2);
                    V = axpby(T(1), V, T(b[0]), I);
                }
                Dense<T> X;
                solve(n, axpby(T(1), V, T(-1), U), axpby(T(1), V, T(1), U), X);   // (V-U)^-1 (V+U)
                for (int i = 0; i < squarings; ++i)
                    X = mul(n, X, X);
                return X;
            }

            /// principal square root by the product form of the Denman-Beavers iteration;
            /// false when it does not converge (an eigenvalue on the closed negative real axis)
            template<typename T>
            auto sqrtm(msize_type n, Dense<T> &A) -> bool {
                const auto I = identity<T>(n);
                Dense<T> M = A, Y = A, Minv;
                float64 prev = std::numeric_limits<float64>::infinity();
                for (int it = 0; it < 100; ++it) {
                    if (!solve(n, M, I, Minv))
                        return false;
                    Y = mul(n, Y, axpby(T(0.5), I, T(0.5), Minv));       // Y (I + M^-1) / 2
                    M = axpby(T(0.5), I, T(0.25), axpby(T(1), M, T(1), Minv));   // (I + (M + M^-1) / 2) / 2
                    const float64 err = norm1(n, axpby(T(1), M, T(-1), I));
                    if (err <= 1e-15 * static_cast<float64>(n) || (err < 1e-10 && err >= prev)) {
                        A = std::move(Y);
                        return true;
                    }
                    prev = err;
                }
                return false;
            }

            enum class LogStatus { ok, singular, no_principal };

            /**
             * principal log(A) by inverse scaling and squaring: square roots until
             * ||A - I||_1 <= 1/4, then log(I + X) by the [8/8] Pade approximant in partial
             * fractions (8-point Gauss-Legendre on log(I + X) = int_0^1 X (I + tX)^-1 dt),
             * times 2^k. See N. J. Higham, "Functions of Matrices", SIAM 2008, ch. 11.
             */
            template<typename T>
            auto logm(msize_type n, Dense<T> A, LogStatus &status) -> Dense<T> {
                const auto I = identity<T>(n);
                Dense<T> scratch;
                status = LogStatus::ok;
                if (!solve(n, A, I, scratch)) {
                    status = LogStatus::singular;
                    return {};
                }
                int k = 0;
                while (!(norm1(n, axpby(T(1), A, T(-1), I)) <= 0.25)) {
                    if (++k > 64 || !sqrtm(n, A)) {
                        status = LogStatus::no_principal;
                        return {};
                    }
                }
                static const float64 node[] = {0.1834346424956498, 0.5255324099163290, 0.7966664774136267,
                                               0.9602898564975363};
                static const float64 weight[] = {0.3626837833783620, 0.3137066458778873, 0.2223810344533745,
                                                 0.1012285362903763};
                const auto X = axpby(T(1), A, T(-1), I);
                Dense<T> L(n * n, T(0)), term;
                for (int j = 0; j < 4; ++j)
                    for (int sgn = -1; sgn <= 1; sgn += 2) {
                        const float64 t = (1 + sgn * node[j]) / 2;   // node on [0, 1]
                        solve(n, axpby(T(1), I, T(t), X), X, term);   // (I + tX)^-1 X
                        L = axpby(T(1), L, T(weight[j] / 2), term);
                    }
                const T scale = T(std::ldexp(1.0, k));
                for (auto &l: L)
                    l *= scale;
                return L;
            }

            /// computation type: complex when the input or the requested result is complex
            template<typename Out, typename In>
            using compute_t = std::conditional_t<complex_type<Out>::value || complex_type<In>::value,
                                                 complex_max, real_max>;

            /// the square input as a Dense matrix (a single element is a 1x1 matrix)
            template<typename C, typename dType>
            auto input(const Matrix<dType> &m, const char *name, msize_type &n) -> Dense<C> {
                const auto &shape = m.getShape();
                if (m.getSize() <= 1)
                    n = m.getSize();
                else if (shape.size() == 2 && shape[0] == shape[1])
                    n = shape[0];
                else
                    throw_value_error(std::string(name) + ": requires a square matrix");
                Dense<C> A(n * n);
                for (msize_type i = 0; i < n * n; ++i)
                    A[i] = C(m.getData()[i]);
                return A;
            }

            template<typename Out, typename C>
            auto output(const Dense<C> &A, const mshape_type &shape) -> Matrix<Out> {
                if (A.empty())
                    return {};
                Matrix<Out> r(shape, Out(0));
                for (size_t i = 0; i < A.size(); ++i)
                    r.getData()[i] = static_cast<Out>(A[i]);
                return r;
            }

            template<typename Out, typename In>
            constexpr void check_types(const char *) {
                static_assert(calculable_type<Out>::value, "matrix function: output type must be Calculable");
                static_assert(complex_type<Out>::value || !complex_type<In>::value,
                              "matrix function: a complex matrix needs a complex result type");
            }
        }

        /**
         * expm(A), the matrix exponential of a square matrix (MATLAB expm; the element-wise
         * exponential is exp). Result: real_max, or complex_max for complex input;
         * expm<complex128>(A) requests a complex result.
         */
        template<typename TargetType = void, typename dType>
        auto expm(const Matrix<dType> &m) -> Matrix<resolve_t<TargetType, linalg_promote_t<dType>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            using C = MatFun::compute_t<Out, dType>;
            MatFun::check_types<Out, dType>("expm");
            msize_type n;
            auto A = MatFun::input<C>(m, "expm", n);
            return MatFun::output<Out>(MatFun::expm(n, std::move(A)), m.getShape());
        }

        /**
         * logm(A), the principal matrix logarithm (MATLAB logm; the element-wise logarithm is
         * log). Needs no eigenvalue on the closed negative real axis: a real matrix with a
         * negative eigenvalue has a complex logarithm, which this implementation (no Schur
         * decomposition) cannot compute, even as logm<complex128>(A).
         */
        template<typename TargetType = void, typename dType>
        auto logm(const Matrix<dType> &m) -> Matrix<resolve_t<TargetType, linalg_promote_t<dType>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<dType>>;
            using C = MatFun::compute_t<Out, dType>;
            MatFun::check_types<Out, dType>("logm");
            msize_type n;
            auto A = MatFun::input<C>(m, "logm", n);
            MatFun::LogStatus status;
            auto L = MatFun::logm(n, std::move(A), status);
            if (status == MatFun::LogStatus::singular)
                throw_value_error("logm: the matrix is singular, its logarithm is not finite");
            if (status == MatFun::LogStatus::no_principal)
                throw_value_error("logm: the matrix has an eigenvalue on the closed negative real axis "
                                  "(the logarithm of a real matrix is then complex, and complex "
                                  "results such as logm<complex128>(A) are not supported for it)");
            return MatFun::output<Out>(L, m.getShape());
        }

        /**
         * mpower(A, p) = A^p, the matrix power (MATLAB mpower; the element-wise power is the
         * ^ operator). Integer p: repeated squaring (A^0 = I; p < 0 inverts A first). Other p:
         * the principal power expm(p * logm(A)).
         */
        template<typename TargetType = void, typename lT, typename rT,
                 std::enable_if_t<std::is_arithmetic_v<rT>, int> = 0>
        auto mpower(const Matrix<lT> &m, const rT &p) -> Matrix<resolve_t<TargetType, linalg_promote_t<lT>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<lT>>;
            using C = MatFun::compute_t<Out, lT>;
            MatFun::check_types<Out, lT>("mpower");
            msize_type n;
            auto A = MatFun::input<C>(m, "mpower", n);
            const auto pf = static_cast<float64>(p);
            if (pf == std::trunc(pf) && std::abs(pf) < 9007199254740992.0) {
                const auto I = MatFun::identity<C>(n);
                auto k = static_cast<int64>(pf);
                if (k < 0) {
                    if (!MatFun::solve(n, A, I, A))
                        throw_value_error("mpower: the matrix is singular, it has no negative power");
                    k = -k;
                }
                auto R = I;
                while (k > 0) {
                    if (k & 1)
                        R = MatFun::mul(n, R, A);
                    k >>= 1;
                    if (k > 0)
                        A = MatFun::mul(n, A, A);
                }
                return MatFun::output<Out>(R, m.getShape());
            }
            MatFun::LogStatus status;
            auto L = MatFun::logm(n, std::move(A), status);
            if (status == MatFun::LogStatus::singular)
                throw_value_error("mpower: the matrix is singular, a non-integer power is not supported");
            if (status == MatFun::LogStatus::no_principal)
                throw_value_error("mpower: the matrix has an eigenvalue on the closed negative real axis, "
                                  "a non-integer power is not supported");
            for (auto &l: L)
                l *= C(pf);
            return MatFun::output<Out>(MatFun::expm(n, std::move(L)), m.getShape());
        }

        /**
         * mpower(s, A) = s^A = expm(log(s) * A) for a scalar base s != 0. A negative base needs
         * a complex result: mpower<complex128>(s, A).
         */
        template<typename TargetType = void, typename lT, typename rT,
                 std::enable_if_t<std::is_arithmetic_v<lT>, int> = 0>
        auto mpower(const lT &s, const Matrix<rT> &m) -> Matrix<resolve_t<TargetType, linalg_promote_t<rT>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<rT>>;
            using C = MatFun::compute_t<Out, rT>;
            MatFun::check_types<Out, rT>("mpower");
            const auto sf = static_cast<float64>(s);
            if (sf == 0)
                throw_value_error("mpower: the base must be nonzero");
            C log_s;
            if constexpr (complex_type<C>::value)
                log_s = std::log(C(sf));
            else {
                if (sf < 0)
                    throw_value_error("mpower: negative base: s^A is complex; request a complex result, "
                                      "e.g. mpower<complex128>(s, A)");
                log_s = std::log(sf);
            }
            msize_type n;
            auto A = MatFun::input<C>(m, "mpower", n);
            for (auto &a: A)
                a *= log_s;
            return MatFun::output<Out>(MatFun::expm(n, std::move(A)), m.getShape());
        }

        /**
         * mpower(A, B) with matrices: one of them must hold a single element (a scalar), as in
         * MATLAB. For an element-wise power use A ^ B.
         */
        template<typename TargetType = void, typename lT, typename rT>
        auto mpower(const Matrix<lT> &A, const Matrix<rT> &B)
                -> Matrix<resolve_t<TargetType, linalg_promote_t<decltype(std::declval<lT>() * std::declval<rT>())>>> {
            using Out = resolve_t<TargetType, linalg_promote_t<decltype(std::declval<lT>() * std::declval<rT>())>>;
            if constexpr (!complex_type<rT>::value) {
                if (B.getSize() == 1)
                    return mpower<Out>(A, B.getData()[0]);
            }
            if constexpr (!complex_type<lT>::value) {
                if (A.getSize() == 1)
                    return mpower<Out>(A.getData()[0], B);
            }
            throw_value_error("mpower: the base or the exponent must be a scalar; for an element-wise "
                              "power use A ^ B");
            return {};
        }
    }

    ///matrix alter
    namespace Math {
        template<typename dType>
        inline auto T(const Matrix<dType> &m) -> Matrix<dType> {
            return m.T();
        }

        template<typename dType, typename TargetType = decltype(matsar::abs(std::declval<dType>()))>
        auto abs(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function(m, [](const dType &x) { return matsar::abs(x); });
        }

//        template<Calculable dType>
//        auto det(const Matrix<dType> &m) -> decltype(m.det()) {
//            return m.det();
//        }

//        template<Calculable dType>
//        auto inv(const Matrix<dType> &m) -> decltype(m.inv()) {
//            return m.inv();
//        }

        template<typename dType>
        inline auto cat(const vector<Matrix<dType>> &mv, const int64 &dim = 1) -> Matrix<dType> {
            return {mv, dim};
        }

        template<typename dType>
        inline auto cat(const initializer_list<Matrix<dType>> &mv, const int64 &dim = 1) -> Matrix<dType> {
            return {mv, dim};
        }
    }

    ///check matrix (iss)
    namespace Math {
        template<typename dType>
        inline auto isSquare(const Matrix<dType> &m) -> bool {
            return m.isSquare();
        }

        template<typename dType>
        inline auto isDiag(const Matrix<dType> &m) -> bool {
            return m.isDiag();
        }

        template<typename dType>
        inline auto isEmpty(const Matrix<dType> &m) -> bool {
            return m.isEmpty();
        }
    }

    ///permute
    namespace Math {
        /**
         * permute matrix according to dimorder
         */
        template<typename dType>
        auto permute(const Matrix<dType> &m, mshape_type dimorder) -> Matrix<dType> {
            if (dimorder.size() != m.getDims()) {
                throw_value_error("permute: dimorder needs to have the same length as the matrix dimensions");
                return {};
            }
            //New Matrix Shape
            vector<bool> pick_check(dimorder.size(), false);
            auto r_shape = m.getShape();
            for (size_t i = 0; i < dimorder.size(); ++i) {
                if (dimorder[i] <= 0 || dimorder[i] > m.getDims()) {
                    throw_value_error("permute: The number in dimorder must be from 1 to n");
                }
                dimorder[i] = dimorder[i] - 1;//trans numeric index to array index
                if (pick_check[dimorder[i]]) {
                    throw_value_error("permute: The numbers in dimorder are repeated");
                }
                pick_check[dimorder[i]] = true;
                if (i != dimorder[i]) {
                    r_shape[i] = m.getShape()[dimorder[i]];
                }
            }
            //New Matrix Data Acquisition
            //312->231
            auto idimorder(dimorder);
            for (size_t i = 0; i < dimorder.size(); ++i) {
                idimorder[dimorder[i]] = msize_type(i);
            }

            Matrix<dType> r(r_shape, 0.);
            mshape_type m_picker(m.getDims(), 0);
            mshape_type r_picker(m.getDims(), 0);

            for (auto &i: m) {
                r(r_picker) = i;
                for (size_t j = 0; j < m_picker.size(); ++j) {
                    m_picker[j]++;
                    if (m_picker[j] == m.getShape()[j]) {
                        m_picker[j] = 0;
                        r_picker[idimorder[j]] = m_picker[j];
                        continue;
                    }
                    r_picker[idimorder[j]] = m_picker[j];
                    break;
                }
            }
            return r;
        }

        /**
         * Inverse swap the dimensions of the matrix using dimorder as the original order
         */
        template<typename dType>
        auto ipermute(const Matrix<dType> &m, mshape_type dimorder) -> Matrix<dType> {
            // Validate here: dimorder[i] - 1 indexes idimorder below, so an out-of-range entry
            // wrote out of bounds and a repeated one built a wrong order that permute accepted.
            if (dimorder.size() != m.getDims())
                throw_value_error("ipermute: dimorder needs to have the same length as the matrix dimensions");
            vector<bool> seen(dimorder.size(), false);
            for (auto d: dimorder) {
                if (d <= 0 || d > m.getDims())
                    throw_value_error("ipermute: The number in dimorder must be from 1 to n");
                if (seen[d - 1])
                    throw_value_error("ipermute: The numbers in dimorder are repeated");
                seen[d - 1] = true;
            }
            auto idimorder(dimorder);
            for (size_t i = 0; i < dimorder.size(); ++i) {
                //trans numeric index to array index -1 +1
                idimorder[dimorder[i] - 1] = msize_type(i + 1);
            }
            return permute(m, idimorder);
        }
    }

    namespace Math {
        /**
         * nnz, The total number of nonzero elements in the matrix
         */
        template<typename dType>
        auto nnz(const Matrix<dType> &m) -> msize_type {
            // Counts directly: `m != 0` does not compile for a const Matrix (Matrix's comparison
            // operators are non-const members) nor for a non-int element type.
            msize_type count = 0;
            for (const auto &x: m)
                if (x != dType(0))
                    ++count;
            return count;
        }

        /**
         * nonzeros, Non-zero elements of the matrix
         * @author:jie
         */
        template<typename dType>
        auto nonzeros(const Matrix<dType> &m) -> Matrix<dType> {
            auto r = zeros<dType>({nnz(m)});
            auto iter_r = r.begin();
            for (const auto &i: m) {
                if (i != 0) {
                    *iter_r = i;
                    iter_r++;
                }
            }
            return r;
        }
    }
}

#endif //MATSAR_MATRIXOPERATION_HPP