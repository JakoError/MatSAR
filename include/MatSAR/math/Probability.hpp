// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2022/7/11.
//

#ifndef MATSAR_PROBABILITY_HPP
#define MATSAR_PROBABILITY_HPP

#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

//#ifndef _USE_MATH_DEFINES
//#define _USE_MATH_DEFINES
//#endif //_USE_MATH_DEFINES

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

#ifndef MATSAR_MATRIXOPERATION_HPP

#include "MatrixOperation.hpp"

#endif //MATSAR_MATRIXOPERATION_HPP

#ifndef MATSAR_MATHEMATICALFUNCTION_HPP

#include "MathematicalFunction.hpp"

#endif //MATSAR_MATHEMATICALFUNCTION_HPP

#ifndef MATSAR_SORTSEARCHCOUNT_HPP

#include "SortSearchCount.hpp"

#endif //MATSAR_SORTSEARCHCOUNT_HPP

namespace matsar {
///Matrix probability computation functions
    namespace Math {
        namespace ProbDetail {
            template<typename T>
            auto is_nan(const T &v) -> bool {
                if constexpr (std::is_floating_point_v<T>)
                    return std::isnan(v);
                else
                    return false;
            }

            /// the "best" element along dim (max: greater, min: less). NaN is skipped, as in MATLAB;
            /// a slice holding only NaN gives NaN.
            template<typename dType, typename Better>
            auto extreme(const Matrix<dType> &m, msize_type dim, Better better) -> Matrix<dType> {
                static_assert(!complex_type<dType>::value, "max / min: complex matrices are not supported");
                auto take = [&better](const dType &best, const dType &v) {
                    return is_nan(best) || (!is_nan(v) && better(v, best));
                };
                if (m.isEmpty())
                    return {};
                auto dim_idx = dim - 1;
                if (dim_idx >= m.getDims())
                    return m;
                if (dim_idx < 0) {
                    dType best = m.getData()[0];
                    for (msize_type i = 1; i < m.getSize(); ++i)
                        if (take(best, m.getData()[i]))
                            best = m.getData()[i];
                    return Matrix<dType>(mshape_type{1}, best);
                }
                auto shape = m.getShape();
                auto step_size = matsar::Util::VectorUtil::mul(shape, 0, dim_idx);
                auto unit_size = step_size * shape[dim_idx];
                auto step_num = shape[dim_idx];
                auto unit_num = m.getSize() / unit_size;

                auto r_shape = m.getShape();
                r_shape[dim_idx] = 1;
                Matrix<dType> r(r_shape, dType(0));

                auto iter_r = r.getData(), iter_m = m.getData();
                for (msize_type i = 0; i < unit_num; ++i) {
                    for (msize_type j = 0; j < step_size; ++j) {
                        *iter_r = *(iter_m + j);
                        for (msize_type k = 1; k < step_num; ++k) {
                            const auto &item = *(iter_m + j + step_size * k);
                            if (take(*iter_r, item))
                                *iter_r = item;
                        }
                        iter_r++;
                    }
                    iter_m += unit_size;
                }
                return r;
            }
        }

        /**
         * Return the maximum along the given dimension of the matrix (NaN is skipped, as in MATLAB)
         * @param m a real-valued matrix
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         * @return the maximum along the given dimension of the matrix
         */
        template<Calculable dType>
        auto max(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<dType> {
            return ProbDetail::extreme(m, dim, [](const dType &a, const dType &b) { return a > b; });
        }

        /**
         * Return the minimum along the given dimension of the matrix (NaN is skipped, as in MATLAB)
         * @param m a real-valued matrix
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         * @return the minimum along the given dimension of the matrix
         */
        template<Calculable dType>
        auto min(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<dType> {
            return ProbDetail::extreme(m, dim, [](const dType &a, const dType &b) { return a < b; });
        }

        /**
         * @related the range function is equivalent to calling the min and max functions
         */
        template<Calculable dType>
        inline auto range(const Matrix<dType> &m, msize_type dim = 1) -> tuple<Matrix<dType>, Matrix<dType>> {
            return {min(m, dim), max(m, dim)};
        }

        /**
         * Return the median along the given dimension of the matrix
         * @param m supports all Calculable matrices
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         * @return the median along the given dimension of the matrix
         */
        template<typename TargetType = void, Calculable dType>
        inline auto median(const Matrix<dType> &m, msize_type dim = 1)
                -> Matrix<resolve_t<TargetType, real_promote_t<dType>>> {
            using Out = resolve_t<TargetType, real_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "median: output type must be Calculable");
            if (m.isEmpty())
                return {};
            auto dim_idx = dim - 1;
            if (dim_idx >= m.getDims())
                return Matrix<Out>(m);
            if (dim_idx < 0) {
                auto sort_m = m.sort();
                auto iter = sort_m.getData() + sort_m.getSize() / 2;
                if (sort_m.getSize() % 2 == 0) {
                    return {(real_promote_t<dType>(*(iter - 1)) + real_promote_t<dType>(*iter)) /
                            real_promote_t<dType>(2)};
                } else {
                    return {real_promote_t<dType>(*iter) / real_promote_t<dType>(1)};
                }
            }
            auto shape = m.getShape();
            auto step_size = matsar::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto sort_m = matsar::Math::sort(m, dim_idx);   // sort takes the 0-based axis
            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<Out> r(r_shape, 0., m.getData());

            auto iter_r = r.getData();
            auto iter_m = sort_m.getData();
            if (shape[dim_idx] % 2 == 0) {
                auto pick = (step_num / 2 - 1) * step_size;
                iter_m += pick;
                for (msize_type i = 0; i < unit_num; ++i) {
                    for (msize_type j = 0; j < step_size; ++j) {
                        *(iter_r++) = (real_promote_t<dType>(*(iter_m + j)) +
                                       real_promote_t<dType>(*(iter_m + j + step_size))) /
                                      real_promote_t<dType>(2);
                    }
                    iter_m += unit_size;
                }
            } else {
                auto pick = (step_num / 2) * step_size;
                iter_m += pick;
                for (msize_type i = 0; i < unit_num; ++i) {
                    for (msize_type j = 0; j < step_size; ++j) {
                        *(iter_r++) = *(iter_m + j);
                    }
                    iter_m += unit_size;
                }
            }
            return r;
        }

        /**
         * mean computes the mean of n values
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         */
        template<typename TargetType = void, Calculable dType>
        inline auto mean(const Matrix<dType> &m, msize_type dim = 1)
                -> Matrix<resolve_t<TargetType, real_promote_t<dType>>> {
            using Out = resolve_t<TargetType, real_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "mean: output type must be Calculable");
            auto dim_idx = dim - 1;
            if (dim_idx >= m.getDims())
                return Matrix<Out>(m);
            // Divide in-place (operator/=) rather than via the free Matrix operator/,
            // which widens the result to arith_promote_t<Out> (float32 -> float64,
            // complex64 -> complex128) for ANY scalar divisor; that wider value is then
            // narrowed back to Out on return, routing through the type-loss-warning
            // constructor and spewing a spurious TypeLossException to stderr even though
            // the numeric result is correct. Staying in Out precision throughout avoids
            // the widen->narrow round trip -- but operator/= independently re-checks
            // typeHigher<Other, Out> on the DIVISOR's own type, so the divisor must not
            // be wrapped in float_max (== float64): float64 tests higher than Out when
            // Out is float32, which would re-trigger the very same warning. The raw
            // integer count (msize_type) is never higher-level than any real/complex
            // Out, so divide by it directly -- the same technique var's inner mean below
            // already relies on (`mean_m /= n`).
            // dim is 1-based (like median/max/min); sum's axis parameter is 0-based
            // and uses the `none` sentinel -- not a numeric value -- for a true global
            // reduce. Convert here: global (dim <= 0) -> sum(m, none) / getSize();
            // axis -> sum(m, dim_idx) / axis length.
            if (dim_idx < 0) {
                auto acc = sum<Out>(m, none);
                acc /= m.getSize();
                return acc;
            }
            auto acc = sum<Out>(m, dim_idx);
            acc /= m.getShape()[dim_idx];
            return acc;
        }
    }
    namespace Math {
        /**
         * variance, as MATLAB var: w = false normalizes by N, w = true by N - 1 (the sample
         * variance); a slice of one value is normalized by N either way, so its variance is 0.
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, every element is a slice of one value: 0, or NaN for NaN / Inf
         */
        template<typename TargetType = void, Calculable dType>
        inline auto var(const Matrix<dType> &m, bool w = false, msize_type dim = 1)
                -> Matrix<resolve_t<TargetType, magnitude_t<dType>>> {
            using Out = resolve_t<TargetType, magnitude_t<dType>>;
            static_assert(calculable_type<Out>::value, "var: output type must be Calculable");
            msize_type dim_idx = dim - 1;
            if (m.isEmpty()) return {};
            if (dim_idx >= m.getDims()) {   // one-value slices: x - mean(x) is 0, or NaN for NaN / Inf
                Matrix<Out> r(m.getShape(), Out(0));
                const dType *src = m.getData();
                Out *dst = r.getData();
                for (msize_type k = 0; k < m.getSize(); ++k) {
                    bool finite;
                    if constexpr (complex_type<dType>::value)
                        finite = std::isfinite(double(src[k].real())) && std::isfinite(double(src[k].imag()));
                    else
                        finite = std::isfinite(double(src[k]));
                    if (!finite)
                        dst[k] = Out(std::numeric_limits<double>::quiet_NaN());
                }
                return r;
            }
            // dim is 1-based (like median/max/min); sum's axis parameter is 0-based and
            // uses the `none` sentinel -- not a numeric value -- for a true global reduce.
            // Both of var's sum calls (the inner mean AND the outer squared-magnitude sum)
            // must therefore pass `dim_idx` for an axis reduce and `none` for a global one,
            // and the element count `n` is the axis length for an axis reduce vs the whole
            // size for a global one.
            //
            // Divide in-place (operator/=) rather than via the free Matrix operator/,
            // which widens complex64 -> complex128 for ANY scalar divisor (arith_promote_t
            // unconditionally promotes complex<float> to complex<double>). That widening
            // would leave m (complex64) minus a complex128 mean with no valid operator-
            // overload (Matrix operator- has no mixed-precision-complex overload). Staying
            // in real_promote_t<dType> precision throughout keeps the subtraction homogeneous.
            // The outer sum uses the same in-place-division technique: dividing via the free
            // operator/ would widen Out (e.g. complex64 -> complex128) then narrow back on
            // return, triggering a spurious TypeLossException.
            if (dim_idx < 0) {
                auto n = m.getSize();
                auto denom = w && n > 1 ? n - 1 : n;   // MATLAB: one value is normalized by N
                auto mean_m = sum<real_promote_t<dType>>(m, none);
                mean_m /= n;
                auto acc = sum<Out>(abs(m - mean_m) ^ 2, none);
                acc /= denom;
                // Return unbraced (like mean): Matrix{acc} routes through the
                // brace constructor, which squeezes trailing singleton dims
                // ({2,1} -> {2}) and would break the keepdims axis shape.
                return acc;
            }
            auto n = m.getShape()[dim_idx];
            auto denom = w && n > 1 ? n - 1 : n;
            auto mean_m = sum<real_promote_t<dType>>(m, dim_idx);
            mean_m /= n;
            auto acc = sum<Out>(abs(m - mean_m) ^ 2, dim_idx);
            acc /= denom;
            return acc;
        }
    }
    ///Distribution functions
    ///Conventions (MATLAB / SciPy): a NaN argument gives NaN; an invalid parameter gives NaN;
    ///x outside the support gives 0 (for the discrete distributions also a non-integer x).
    ///Matrix arguments broadcast against each other like Matrix arithmetic.
    namespace Math {
        namespace ProbDetail {
            inline auto is_integer(float64 v) -> bool { return std::isfinite(v) && v == std::floor(v); }

            /// the common shape of broadcasting arguments (trailing singleton dims appended, as Matrix does)
            inline auto broadcast_shape(const char *name, std::initializer_list<mshape_type> shapes) -> mshape_type {
                size_t dims = 0;
                for (const auto &s: shapes)
                    dims = std::max(dims, s.size());
                mshape_type out(dims, 1);
                for (auto s: shapes) {
                    s.resize(dims, 1);
                    for (size_t i = 0; i < dims; ++i) {
                        if (s[i] == out[i] || s[i] == 1)
                            continue;
                        if (out[i] == 1)
                            out[i] = s[i];
                        else
                            throw_shape_exception(std::string(name) + ": the arguments cannot be broadcast to one shape");
                    }
                }
                return out;
            }

            template<typename F, typename... Ms>
            auto map_expanded(F f, const mshape_type &shape, const Ms &...ms) -> Matrix<float64> {
                Matrix<float64> r(shape, 0.);
                for (msize_type i = 0; i < r.getSize(); ++i)
                    r.getData()[i] = f(ms.getData()[i]...);
                return r;
            }

            /// f applied element by element to broadcast Matrix<float64> arguments
            template<typename F, typename... Ms>
            auto map(const char *name, F f, const Ms &...ms) -> Matrix<float64> {
                const auto shape = broadcast_shape(name, {ms.getShape()...});
                auto expand = [&shape](const Matrix<float64> &m) { return m.getShape() == shape ? m : m.broadcast(shape); };
                return map_expanded(f, shape, expand(ms)...);
            }

            // Binomial, Poisson and gamma densities by C. Loader's saddle-point expansion ("Fast and
            // accurate computation of binomial probabilities", 2000; also R's dbinom / dpois): the
            // density is exp(-(Stirling-series errors) - (deviance terms)), sums of small terms with no
            // large cancellation, so it stays accurate for any n where n! / p^x would over- or underflow.

            /// log(n!) - log(sqrt(2 pi n) (n / e)^n), n > 0
            inline auto stirlerr(float64 n) -> float64 {
                if (n <= 15)
                    return std::lgamma(n + 1) - (n + 0.5) * std::log(n) + n - 0.5 * std::log(2 * PI);
                const float64 nn = n * n;
                const float64 s0 = 1. / 12, s1 = 1. / 360, s2 = 1. / 1260, s3 = 1. / 1680, s4 = 1. / 1188;
                if (n > 500)
                    return (s0 - s1 / nn) / n;
                if (n > 80)
                    return (s0 - (s1 - s2 / nn) / nn) / n;
                if (n > 35)
                    return (s0 - (s1 - (s2 - s3 / nn) / nn) / nn) / n;
                return (s0 - (s1 - (s2 - (s3 - s4 / nn) / nn) / nn) / nn) / n;
            }

            /// the deviance term x log(x / np) + np - x, by its series when x is close to np
            inline auto bd0(float64 x, float64 np) -> float64 {
                if (std::abs(x - np) < 0.1 * (x + np)) {
                    float64 v = (x - np) / (x + np);
                    float64 s = (x - np) * v;
                    float64 ej = 2 * x * v;
                    v *= v;
                    for (int j = 1; j < 1000; ++j) {
                        ej *= v;
                        const float64 s1 = s + ej / (2 * j + 1);
                        if (s1 == s)
                            return s1;
                        s = s1;
                    }
                }
                return x * std::log(x / np) + np - x;
            }

            /// binomial probability of x successes in n trials; integers 0 <= x <= n, 0 <= p <= 1
            inline auto dbinom(float64 x, float64 n, float64 p) -> float64 {
                const float64 q = 1 - p;
                if (p == 0)
                    return x == 0 ? 1 : 0;
                if (q == 0)
                    return x == n ? 1 : 0;
                if (x == 0) {
                    if (n == 0)
                        return 1;
                    return std::exp(p < 0.1 ? -bd0(n, n * q) - n * p : n * std::log(q));
                }
                if (x == n)
                    return std::exp(q < 0.1 ? -bd0(n, n * p) - n * q : n * std::log(p));
                const float64 lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
                const float64 lf = std::log(2 * PI) + std::log(x) + std::log1p(-x / n);
                return std::exp(lc - 0.5 * lf);
            }

            /// Poisson density at x >= 0 (any real x: also the gamma density kernel), lambda >= 0
            inline auto dpois(float64 x, float64 lambda) -> float64 {
                if (lambda == 0)
                    return x == 0 ? 1 : 0;
                if (!std::isfinite(lambda))
                    return 0;
                if (x == 0)
                    return std::exp(-lambda);
                if (x <= lambda * std::numeric_limits<float64>::min())
                    return std::exp(-lambda);
                if (lambda < x * std::numeric_limits<float64>::min())
                    return std::exp(-lambda + x * std::log(lambda) - std::lgamma(x + 1));
                return std::exp(-stirlerr(x) - bd0(x, lambda)) / std::sqrt(2 * PI * x);
            }

            /// gamma density, x > 0 finite, shape > 0, scale > 0
            inline auto dgamma(float64 x, float64 shape, float64 scale) -> float64 {
                if (shape < 1)
                    return dpois(shape, x / scale) * shape / x;
                return dpois(shape - 1, x / scale) / scale;
            }
        }

        /**
         * Normal density, element by element
         * @param m must be a real-valued matrix
         * @param mu the mean of the normal distribution
         * @param sigma the standard deviation, > 0 (otherwise NaN)
         * @return floating-point matrix: the density at each element
         */
        template<RealType RT>
        auto normpdf(const Matrix<RT> &m, float_max mu = 0., float_max sigma = 1.) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            const auto norm = sigma * std::sqrt(2 * PI);
            for (auto &i: r) {
                if (!(sigma > 0)) {
                    i = NAN;
                    continue;
                }
                auto z = (i - mu) / sigma;
                i = std::exp(-z * z / 2.) / norm;
            }
            return r;
        }

        /**
         * Normal density with parameter matrices (broadcast against m)
         */
        template<RealType RT>
        auto normpdf(const Matrix<RT> &m, const Matrix<float_max> &mu, const Matrix<float_max> &sigma)
                -> Matrix<real_max> {
            return ProbDetail::map("normpdf", [](float64 x, float64 mu_, float64 sigma_) -> float64 {
                if (!(sigma_ > 0))
                    return NAN;
                auto z = (x - mu_) / sigma_;
                return std::exp(-z * z / 2.) / (sigma_ * std::sqrt(2 * PI));
            }, Matrix<float64>(m), mu, sigma);
        }

        /**
         * Random matrix from the normal distribution
         * @param sigma the standard deviation: 0 gives mu exactly, < 0 (or NaN) gives NaN
         */
        inline auto normrnd(float_max mu = 0., float_max sigma = 1., const mshape_type &shape = {1}) -> Matrix<real_max> {
            if (!(sigma >= 0) || std::isnan(mu))
                return Matrix<real_max>(shape, NAN);
            if (sigma == 0)
                return Matrix<real_max>(shape, mu);
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            normal_distribution<float_max> normal(mu, sigma);
            for (auto &i: r)
                i = normal(gen);
            return r;
        }

        /**
         * Random matrix from the normal distribution, one draw per element of mu and sigma
         * @param shape the result shape; mu and sigma must have it (empty: the shape of mu)
         */
        inline auto normrnd(const Matrix<float_max> &mu, const Matrix<float_max> &sigma,
                            mshape_type shape = {}) -> Matrix<real_max> {
            if (shape.empty())
                shape = mu.getShape();
            if (mu.getShape() != shape || sigma.getShape() != shape)
                throw_shape_exception("normrnd: mu and sigma must have the requested shape");
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            for (msize_type i = 0; i < r.getSize(); ++i) {
                const auto m = mu.getData()[i], s = sigma.getData()[i];
                if (!(s >= 0) || std::isnan(m))
                    r.getData()[i] = NAN;
                else if (s == 0)
                    r.getData()[i] = m;
                else
                    r.getData()[i] = normal_distribution<float_max>(m, s)(gen);
            }
            return r;
        }

        /**
         * The mean and variance of the normal distribution: {mu, sigma^2}; NaN for sigma < 0
         */
        template<RealType RT1, RealType RT2>
        auto normstat(RT1 mu, RT2 sigma) -> tuple<real_max, real_max> {
            if (sigma < 0)
                return {NAN, NAN};
            return {real_max(mu), real_max(sigma) * real_max(sigma)};
        }

        /**
         * The mean and variance of the normal distribution, for broadcast parameter matrices
         */
        template<RealType RT1, RealType RT2>
        auto normstat(const Matrix<RT1> &mu, const Matrix<RT2> &sigma) -> tuple<Matrix<real_max>, Matrix<real_max>> {
            Matrix<float64> m(mu), s(sigma);
            return {ProbDetail::map("normstat", [](float64 m_, float64 s_) { return std::get<0>(normstat(m_, s_)); }, m, s),
                    ProbDetail::map("normstat", [](float64 m_, float64 s_) { return std::get<1>(normstat(m_, s_)); }, m, s)};
        }
    }
    namespace Math {
        /**
         * Binomial probability of x successes in n trials with success probability p
         * (0 for x outside 0..n or not an integer; NaN unless n is a non-negative integer and
         * 0 <= p <= 1).
         */
        inline auto binopdf(float64 x, float64 n, float64 p) -> float64 {
            if (std::isnan(x) || !ProbDetail::is_integer(n) || n < 0 || !(p >= 0 && p <= 1))
                return NAN;
            if (!ProbDetail::is_integer(x) || x < 0 || x > n)
                return 0;
            return ProbDetail::dbinom(x, n, p);
        }

        /**
         * Binomial probability of each element of m
         * @param m real-valued matrix of success counts
         */
        template<RealType RT>
        auto binopdf(const Matrix<RT> &m, float64 n, float64 p) -> Matrix<float64> {
            Matrix<float64> r(m);
            for (auto &i: r)
                i = binopdf(i, n, p);
            return r;
        }

        /**
         * Binomial probability with parameter matrices (broadcast against m)
         */
        template<RealType RT>
        auto binopdf(const Matrix<RT> &m, const Matrix<float64> &n, const Matrix<float64> &p) -> Matrix<float64> {
            return ProbDetail::map("binopdf", [](float64 x, float64 n_, float64 p_) { return binopdf(x, n_, p_); },
                                   Matrix<float64>(m), n, p);
        }

        namespace ProbDetail {
            inline auto binornd_one(float64 n, float64 p, std::mt19937 &gen) -> float64 {
                if (!is_integer(n) || n < 0 || !(p >= 0 && p <= 1))
                    return NAN;
                return static_cast<float64>(std::binomial_distribution<int64>(static_cast<int64>(n), p)(gen));
            }
        }

        /**
         * Random matrix from the binomial distribution (NaN for invalid n or p)
         */
        inline auto binornd(float64 n = 1, float64 p = 0.5, const mshape_type &shape = {1}) -> Matrix<real_max> {
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            for (auto &i: r)
                i = ProbDetail::binornd_one(n, p, gen);
            return r;
        }

        /**
         * Random matrix from the binomial distribution, one draw per element of n and p
         * (same shapes; NaN for invalid elements)
         */
        inline auto binornd(const Matrix<float_max> &n, const Matrix<float_max> &p) -> Matrix<real_max> {
            if (n.getShape() != p.getShape())
                throw_shape_exception("binornd: n and p have different sizes");
            Matrix<real_max> r(n.getShape(), 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            for (msize_type i = 0; i < r.getSize(); ++i)
                r.getData()[i] = ProbDetail::binornd_one(n.getData()[i], p.getData()[i], gen);
            return r;
        }

        /**
         * As binornd(n, p), with the result shape stated (n and p must have it)
         */
        inline auto binornd(const Matrix<float_max> &n, const Matrix<float_max> &p,
                            const mshape_type &shape) -> Matrix<real_max> {
            if (n.getShape() != shape || p.getShape() != shape)
                throw_shape_exception("binornd: n and p must have the requested shape");
            return binornd(n, p);
        }

        /**
         * The mean and variance of the binomial distribution: {n p, n p (1 - p)}; NaN for invalid n or p
         */
        template<RealType RT1, RealType RT2>
        auto binostat(RT1 n, RT2 p) -> tuple<real_max, real_max> {
            if (!ProbDetail::is_integer(n) || n < 0 || !(p >= 0 && p <= 1))
                return {NAN, NAN};
            auto mu = real_max(n) * real_max(p);
            return {mu, mu * (1 - real_max(p))};
        }

        /**
         * The mean and variance of the binomial distribution, for broadcast parameter matrices
         */
        template<RealType RT1, RealType RT2>
        auto binostat(const Matrix<RT1> &n, const Matrix<RT2> &p) -> tuple<Matrix<real_max>, Matrix<real_max>> {
            Matrix<float64> n_(n), p_(p);
            return {ProbDetail::map("binostat", [](float64 a, float64 b) { return std::get<0>(binostat(a, b)); }, n_, p_),
                    ProbDetail::map("binostat", [](float64 a, float64 b) { return std::get<1>(binostat(a, b)); }, n_, p_)};
        }
    }
    namespace Math {
        //Continuous uniform distribution
        /**
         * Density of the continuous uniform distribution on [a, b]: 1 / (b - a) inside, 0 outside,
         * NaN unless a < b
         */
        inline auto unifpdf(float64 x, float64 a = 0, float64 b = 1) -> float64 {
            if (std::isnan(x) || !(a < b))
                return NAN;
            return (x < a || x > b) ? 0 : 1 / (b - a);
        }

        /**
         * Uniform density with matrix arguments (broadcast against each other)
         */
        inline auto unifpdf(const Matrix<float64> &x, const Matrix<float64> &a = {0},
                            const Matrix<float64> &b = {1}) -> Matrix<float64> {
            return ProbDetail::map("unifpdf", [](float64 x_, float64 a_, float64 b_) { return unifpdf(x_, a_, b_); },
                                   x, a, b);
        }

        /**
        * The mean and variance of the continuous uniform distribution: {(a + b) / 2, (b - a)^2 / 12};
        * NaN unless a < b
        */
        inline auto unifstat(float64 a, float64 b) -> tuple<float64, float64> {
            if (!(a < b))
                return {NAN, NAN};
            return {(a + b) / 2, (b - a) * (b - a) / 12};
        }

        /**
        * The mean and variance of the continuous uniform distribution, for broadcast matrices
        */
        inline auto unifstat(const Matrix<float64> &a, const Matrix<float64> &b) -> tuple<Matrix<float64>, Matrix<float64>> {
            return {ProbDetail::map("unifstat", [](float64 a_, float64 b_) { return std::get<0>(unifstat(a_, b_)); }, a, b),
                    ProbDetail::map("unifstat", [](float64 a_, float64 b_) { return std::get<1>(unifstat(a_, b_)); }, a, b)};
        }
    }

    namespace Math {
        /**
         * Generate a random matrix from the (0,1) uniform distribution
         * @param shape the dimensions of the target matrix
         * @return floating-point matrix: a random matrix from the (0,1) uniform distribution
         */
        inline auto rand(const mshape_type &shape) -> Matrix<real_max> {
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<real_max> uniform(0, 1);
            for (auto &i: r)
                i = uniform(gen);
            return r;
        }

        /**
         * Generate a matrix from the (0,1) uniform distribution
         * @param n indicates that an n*n matrix is generated
         * @return floating-point matrix: a random matrix from the (0,1) uniform distribution
         */
        inline auto rand(msize_type n) -> Matrix<real_max> {
            return rand({n, n});
        }

        /**
         * Generate a matrix from the [imin,imax] uniform distribution
         * @param imin the minimum value of the uniform distribution
         * @param imax the maximum value of the uniform distribution
         * @param shape the shape of the target matrix
         * @return integer matrix: a random matrix from the [imin,imax] uniform distribution
         */
        inline auto randi(int64 imin, int64 imax, const mshape_type &shape) -> Matrix<int64> {
            if (imin > imax)
                throw_value_error("randi requires the minimum of the integer uniform distribution to be less than or equal to the maximum");
            Matrix<int64> r(shape, int64(0));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int64> uniform(imin, imax);
            for (auto &i: r)
                i = uniform(gen);
            return r;
        }

        /**
         * Generate a matrix from the [1,imax] uniform distribution
         * @param imax the maximum value of the uniform distribution
         * @param shape the shape of the target matrix
         * @return integer matrix: a matrix from the [1,imax] uniform distribution
         */
        inline auto randi(int64 imax, const mshape_type &shape) -> Matrix<int64> {
            return randi(1, imax, shape);
        }

        /**
         * Generate a matrix from the [imin,imax] uniform distribution
         * @param n indicates that an n*n matrix is generated
         */
        inline auto randi(int64 imin, int64 imax, msize_type n) -> Matrix<int64> {
            return randi(imin, imax, {n, n});
        }

        /**
         * Generate a matrix from the [1,imax] uniform distribution
         * @param n indicates that an n*n matrix is generated
         */
        inline auto randi(int64 imax, msize_type n) -> Matrix<int64> {
            return randi(1, imax, {n, n});
        }

        /**
         * Generate a standard normal distribution matrix
         * @param shape the shape of the target matrix
         */
        inline auto randn(const mshape_type &shape) -> Matrix<real_max> {
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<real_max> normal(0, 1);
            for (auto &i: r)
                i = normal(gen);
            return r;
        }

        /**
         * Generate a standard normal distribution matrix
         * @param n indicates that an n*n matrix is generated
         */
        inline auto randn(int64 n) -> Matrix<real_max> {
            return randn({n, n});
        }

        /**
         * k distinct integers drawn at random from 1..n, in random order (a partial Fisher-Yates
         * shuffle: every ordered selection is equally likely)
         */
        template<IntType IT>
        auto randperm(IT n, IT k) -> Matrix<IT> {
            if (n < 0) throw_value_error("randperm: n must be non-negative");
            if (k < 0 || k > n) throw_value_error("randperm: k must be between 0 and n");
            std::vector<IT> values(static_cast<size_t>(n));
            std::iota(values.begin(), values.end(), IT(1));
            std::random_device rd;
            std::mt19937_64 gen(rd());
            for (int64 i = 0; i < static_cast<int64>(k); ++i) {
                std::uniform_int_distribution<int64> pick(i, static_cast<int64>(n) - 1);
                std::swap(values[static_cast<size_t>(i)], values[static_cast<size_t>(pick(gen))]);
            }
            Matrix<IT> r(mshape_type{static_cast<msize_type>(k)}, IT(0));
            for (msize_type i = 0; i < r.getSize(); ++i)
                r(i) = values[static_cast<size_t>(i)];
            return r;
        }

        /**
         * A random permutation of the integers 1..n
         */
        template<IntType IT>
        auto randperm(IT n) -> Matrix<IT> {
            return randperm(n, n);
        }
    }

    namespace Math {
        /**
         * Density of the chi-squared distribution with nu degrees of freedom (nu > 0, otherwise
         * NaN), as the gamma density with shape nu / 2 and scale 2
         */
        inline auto chi2pdf(float64 x, float64 nu) -> float64 {
            if (std::isnan(x) || !(nu > 0))
                return NAN;
            if (x < 0 || std::isinf(x))
                return 0;
            if (x == 0)
                return nu < 2 ? INFINITY : (nu == 2 ? 0.5 : 0);
            return ProbDetail::dgamma(x, nu / 2, 2);
        }

        /**
         * Chi-squared density with matrix arguments (broadcast against each other)
         */
        inline auto chi2pdf(const Matrix<float64> &x, const Matrix<float64> &nu) -> Matrix<float64> {
            return ProbDetail::map("chi2pdf", [](float64 x_, float64 nu_) { return chi2pdf(x_, nu_); }, x, nu);
        }
    }
    namespace Math {
        //Exponential distribution
        /**
         * Density of the exponential distribution with mean mu (> 0, otherwise NaN); 0 for x < 0
         */
        inline auto exppdf(float64 x, float64 mu = 1) -> float64 {
            if (std::isnan(x) || !(mu > 0))
                return NAN;
            if (x < 0)
                return 0;
            return std::exp(-x / mu) / mu;
        }

        /**
         * Exponential density with matrix arguments (broadcast against each other)
         */
        inline auto exppdf(const Matrix<float64> &x, const Matrix<float64> &mu = {1}) -> Matrix<float64> {
            return ProbDetail::map("exppdf", [](float64 x_, float64 mu_) { return exppdf(x_, mu_); }, x, mu);
        }

        /**
         * Random matrix from the exponential distribution with mean mu (0 gives zeros; < 0 or NaN gives NaN)
         */
        inline auto exprnd(float64 mu, const mshape_type &shape = {1}) -> Matrix<float64> {
            if (!(mu >= 0))
                return Matrix<float64>(shape, NAN);
            if (mu == 0)
                return Matrix<float64>(shape, 0.);
            Matrix<float64> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::exponential_distribution<float64> exp_dis(1 / mu);   // std takes the rate, 1/mean
            for (auto &i: r)
                i = exp_dis(gen);
            return r;
        }

        /**
         * The mean and variance of the exponential distribution: {mu, mu^2}; NaN for mu < 0
         */
        template<RealType RT>
        auto expstat(const RT &mu) -> tuple<real_max, real_max> {
            real_max tmp = mu;
            if (tmp < 0)
                return {NAN, NAN};
            return {tmp, tmp * tmp};
        }

        /**
         * The mean and variance of the exponential distribution, element by element
         */
        template<RealType RT>
        auto expstat(const Matrix<RT> &mu) -> tuple<Matrix<real_max>, Matrix<real_max>> {
            Matrix<float64> m(mu);
            return {ProbDetail::map("expstat", [](float64 v) { return std::get<0>(expstat(v)); }, m),
                    ProbDetail::map("expstat", [](float64 v) { return std::get<1>(expstat(v)); }, m)};
        }
    }

    namespace Math {
        //Poisson distribution
        /**
         * Poisson probability of x events at rate lambda (>= 0, otherwise NaN); 0 for x < 0 or
         * not an integer
         */
        inline auto poisspdf(float_max x, float_max lambda) -> float_max {
            if (std::isnan(x) || !(lambda >= 0))
                return NAN;
            if (!ProbDetail::is_integer(x) || x < 0)
                return 0;
            return ProbDetail::dpois(x, lambda);
        }

        /**
         * Poisson probability with matrix arguments (broadcast against each other)
         */
        inline auto poisspdf(const Matrix<float64> &x, const Matrix<float64> &lambda) -> Matrix<float64> {
            return ProbDetail::map("poisspdf", [](float64 x_, float64 l_) { return poisspdf(x_, l_); }, x, lambda);
        }

        /**
         * Random matrix from the Poisson distribution (lambda 0 gives zeros; < 0 or NaN gives NaN)
         */
        inline auto poissrnd(float64 lambda, const mshape_type &shape = {1}) -> Matrix<float64> {
            if (!(lambda >= 0))
                return Matrix<float64>(shape, NAN);
            if (lambda == 0)
                return Matrix<float64>(shape, 0.);
            Matrix<float64> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::poisson_distribution<int64> poisson(lambda);
            for (auto &i: r)
                i = static_cast<float64>(poisson(gen));
            return r;
        }
    }
    namespace Math {
        //Rayleigh distribution
        /**
         * Density of the Rayleigh distribution with scale b (> 0, otherwise NaN); 0 for x < 0
         */
        inline auto raylpdf(float64 x, float64 b) -> float64 {
            if (std::isnan(x) || !(b > 0))
                return NAN;
            if (x < 0 || std::isinf(x))
                return 0;
            auto b2 = b * b;
            return x / b2 * std::exp(-x * x / 2 / b2);
        }

        /**
         * Rayleigh density with matrix arguments (broadcast against each other)
         */
        inline auto raylpdf(const Matrix<float64> &x, const Matrix<float64> &b) -> Matrix<float64> {
            return ProbDetail::map("raylpdf", [](float64 x_, float64 b_) { return raylpdf(x_, b_); }, x, b);
        }
    }
    namespace Math {
        //Weibull distribution
        /**
         * Density of the Weibull distribution with scale lambda and shape k (both > 0, otherwise
         * NaN): k / lambda (x / lambda)^(k - 1) exp(-(x / lambda)^k); 0 for x < 0. At x = 0 it is
         * Inf for k < 1 and 1 / lambda for k = 1 (the exponent is taken separately: the old
         * exp(-(t^(k-1)) * t) was exp(-Inf * 0) = NaN there).
         */
        inline auto weibpdf(float64 x, float64 lambda, float64 k) -> float64 {
            if (std::isnan(x) || !(lambda > 0) || !(k > 0))
                return NAN;
            if (x < 0 || std::isinf(x))
                return 0;
            const auto t = x / lambda;
            return k / lambda * std::pow(t, k - 1) * std::exp(-std::pow(t, k));
        }

        /**
         * Weibull density with matrix arguments (broadcast against each other)
         */
        inline auto weibpdf(const Matrix<float64> &x, const Matrix<float64> &lambda, const Matrix<float64> &k)
                -> Matrix<float64> {
            return ProbDetail::map("weibpdf", [](float64 x_, float64 l_, float64 k_) { return weibpdf(x_, l_, k_); },
                                   x, lambda, k);
        }
    }
}

#endif //MATSAR_PROBABILITY_HPP
