// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2022/7/11.
//

#ifndef MATSAR_PROBABILITY_HPP
#define MATSAR_PROBABILITY_HPP

#include <random>
#include <algorithm>

//#ifndef _USE_MATH_DEFINES
//#define _USE_MATH_DEFINES
//#endif //_USE_MATH_DEFINES

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

#ifndef MATSAR_MATRIXOPERATION_HPP

#include "MatrixOperation.hpp"

#endif //MATSAR_MATRIXOPERATION_HPP

#ifndef MATSAR_LINEARALGEBRA_HPP

#include "LinearAlgebra.hpp"

#endif //MATSAR_LINEARALGEBRA_HPP

#ifndef MATSAR_SORTSEARCHCOUNT_HPP

#include "SortSearchCount.hpp"

#endif //MATSAR_SORTSEARCHCOUNT_HPP

namespace matsar {
///Matrix probability computation functions
    namespace Math {
        /**
         * Return the maximum along the given dimension of the matrix
         * @param m supports all Calculable matrices
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         * @return the maximum along the given dimension of the matrix
         */
        template<Calculable dType>
        auto max(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<dType> {
            auto dim_idx = dim - 1;
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0)
                return {m.max()};
            auto shape = m.getShape();
            auto step_size = matsar::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<dType> r(r_shape, 0., m.getData());

            auto iter_r = r.getData(), iter_m = m.getData();
            for (msize_type i = 0; i < unit_num; ++i) {
                for (msize_type j = 0; j < step_size; ++j) {
                    *iter_r = *(iter_m + j);
                    for (msize_type k = 1; k < step_num; ++k) {
                        auto &item = *(iter_m + j + step_size * k);
                        if (item > *iter_r)
                            *iter_r = item;
                    }
                    iter_r++;
                }
                iter_m += unit_size;
            }
            return r;
        }

        /**
         * Return the minimum along the given dimension of the matrix
         * @param m supports all Calculable matrices
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         * @return the minimum along the given dimension of the matrix
         */
        template<Calculable dType>
        auto min(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<dType> {
            auto dim_idx = dim - 1;
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0)
                return {m.min()};
            auto shape = m.getShape();
            auto step_size = matsar::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<dType> r(r_shape, 0., m.getData());

            auto iter_r = r.getData(), iter_m = m.getData();
            for (msize_type i = 0; i < unit_num; ++i) {
                for (msize_type j = 0; j < step_size; ++j) {
                    *iter_r = *(iter_m + j);
                    for (msize_type k = 1; k < step_num; ++k) {
                        auto &item = *(iter_m + j + step_size * k);
                        if (item < *iter_r)
                            *iter_r = item;
                    }
                    iter_r++;
                }
                iter_m += unit_size;
            }
            return r;
        }

        /**
         * @related the range function is equivalent to calling the min and max functions
         */
        template<Calculable dType>
        auto range(const Matrix<dType> &m, msize_type dim = 1) -> tuple<Matrix<dType>, Matrix<dType>> {
            return {min(m, dim), max(m, dim)};
        }

        /**
         * Return the median along the given dimension of the matrix
         * @param m supports all Calculable matrices
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         * @return the median along the given dimension of the matrix
         */
        template<typename TargetType = void, Calculable dType>
        auto median(const Matrix<dType> &m, msize_type dim = 1)
                -> Matrix<resolve_t<TargetType, real_promote_t<dType>>> {
            using Out = resolve_t<TargetType, real_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "median: output type must be Calculable");
            auto dim_idx = dim - 1;
            if (dim_idx >= m.getDims())
                return {m};
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

            auto sort_m = matsar::Math::sort(m, dim);
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
        auto mean(const Matrix<dType> &m, msize_type dim = 1)
                -> Matrix<resolve_t<TargetType, real_promote_t<dType>>> {
            using Out = resolve_t<TargetType, real_promote_t<dType>>;
            static_assert(calculable_type<Out>::value, "mean: output type must be Calculable");
            auto dim_idx = dim - 1;
            if (dim_idx >= m.getDims())
                return {m};
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
         * @param dim when negative or 0, evaluate globally; within the dimension range, return the evaluation along that dimension; when it exceeds the number of dimensions, return the matrix itself
         */
        template<typename TargetType = void, Calculable dType>
        auto var(const Matrix<dType> &m, bool w = false, msize_type dim = 1)
                -> Matrix<resolve_t<TargetType, magnitude_t<dType>>> {
            using Out = resolve_t<TargetType, magnitude_t<dType>>;
            static_assert(calculable_type<Out>::value, "var: output type must be Calculable");
            msize_type dim_idx = dim - 1;
            if (m.isEmpty()) return {};
            if (dim_idx >= m.getDims()) {
                // magnitude_t collapses complex dType to a real Out, so the identity
                // "return the matrix itself" has no type-preserving meaning here; fall
                // back to the magnitude (abs) for complex inputs, keep the exact prior
                // identity behavior for real inputs (Out == dType there).
                if constexpr (complex_type<dType>::value) {
                    return {abs(m)};
                } else {
                    return {m};
                }
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
                auto denom = w ? n - 1 : n;
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
            auto denom = w ? n - 1 : n;
            auto mean_m = sum<real_promote_t<dType>>(m, dim_idx);
            mean_m /= n;
            auto acc = sum<Out>(abs(m - mean_m) ^ 2, dim_idx);
            acc /= denom;
            return acc;
        }
    }
    ///Distribution functions
    namespace Math {
//        static const float_max PI = static_cast<float_max>(3.14159265358979323846);

        /**
         * Map every element of the matrix through the normal distribution
         * @param m must be a real-valued matrix
         * @param mu the mean of the normal distribution
         * @param sigma the standard deviation of the normal distribution
         * @return floating-point matrix: the matrix elements mapped through the normal distribution
         */
        template<RealType RT>
        auto normpdf(const Matrix<RT> &m, float_max mu = 0., float_max sigma = 1.) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            auto stat1 = sigma * std::sqrt(2 * PI);
            for (auto &i: r) {
                auto stat2 = (i - mu) / sigma;
                i = std::exp(-stat2 * stat2 / 2.) / stat1;
            }
            return r;
        }

        /**
         * Map every element of the matrix through the normal distribution
         * @param m the matrix must be a real-valued matrix
         * @param mu the mean of the normal distribution; a broadcast operation is applied to the mu matrix
         * @param sigma the standard deviation of the normal distribution; a broadcast operation is applied to the sigma matrix
         * @return floating-point matrix: the matrix elements mapped through the normal distribution
         */
        template<RealType RT>
        auto normpdf(const Matrix<RT> &m, Matrix<float_max> mu, Matrix<float_max> sigma) -> Matrix<real_max> {
            mu = mu.broadcast(m.getShape());
            sigma = sigma.broadcast(m.getShape());
            Matrix<real_max> r(m);
            auto stat1 = std::sqrt(2 * PI);
            for (auto iter = r.begin(), iter_mu = mu.begin(), iter_sigma = sigma.begin();
                 iter != r.end(); ++iter, ++iter_mu, ++iter_sigma) {
                auto stat2 = (*iter - *iter_mu) / *iter_sigma;
                *iter = std::exp(-stat2 * stat2 / 2.) / (stat1 * *iter_sigma);
            }
            return r;
        }

        /**
         * Generate a random matrix from the normal distribution
         * @param mu the mean of the normal distribution
         * @param sigma the standard deviation of the normal distribution
         * @param shape the shape of the target matrix
         * @return floating-point matrix: a random matrix from the normal distribution
         */
        auto normrnd(float_max mu = 0., float_max sigma = 1., const mshape_type &shape = {1}) -> Matrix<real_max> {
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            normal_distribution<float_max> normal(mu, sigma);
            for (auto &i: r)
                i = normal(gen);
            return r;
        }

        /**
         * Generate a random matrix from the normal distribution
         * @param mu the mean of the normal distribution; a broadcast operation is applied to the mu matrix
         * @param sigma the standard deviation of the normal distribution; a broadcast operation is applied to the sigma matrix
         * @param shape the shape of the target matrix
         * @return floating-point matrix: a random matrix from the normal distribution
         */
        auto normrnd(const Matrix<float_max> &mu, const Matrix<float_max> &sigma,
                     const mshape_type &shape = {1}) -> Matrix<real_max> {
            if (mu.getShape() != shape || sigma.getShape() != shape)
                throw_shape_exception("mu matrix or sigma matrix size differs from the specified shape");
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            for (auto iter = r.begin(), iter_mu = mu.begin(), iter_sigma = sigma.begin();
                 iter != r.end(); ++iter, ++iter_mu, ++iter_sigma) {
                normal_distribution<float_max> normal(*iter_mu, *iter_sigma);
                *iter = normal(gen);
            }
            return r;
        }

        /**
         * The mean and variance of the normal distribution
         * @return floating-point matrix tuple {mu,sigma*sigma}
         */
        template<RealType RT1, RealType RT2>
        auto normstat(RT1 mu, RT2 sigma) -> tuple<real_max, real_max> {
            return {mu, mul(sigma, sigma)};
        }

        /**
         * The mean and variance of the normal distribution
         * @return floating-point matrix tuple {mu,sigma*sigma}
         */
        template<RealType RT1, RealType RT2>
        auto normstat(const Matrix<RT1> &mu, const Matrix<RT2> &sigma) -> tuple<Matrix<real_max>, Matrix<real_max>> {
            Matrix<real_max> b_mu(mu);
            Matrix<real_max> b_sigma(sigma);
            Matrix<real_max>::broadcast(mu, sigma);
            return {b_mu, mul(b_sigma, b_sigma)};
        }
    }
    namespace Math {
        /**
         * Map every element of the matrix through the binomial distribution
         * @param m must be a real-valued matrix; floating-point values are rounded
         * @param n the number of trials of the binomial distribution
         * @param p the success probability of the binomial distribution
         * @return floating-point matrix: the matrix elements mapped through the binomial distribution
         */
        template<RealType RT>
        auto binopdf(const Matrix<RT> &m, float64 n, float64 p) -> Matrix<float64> {
            Matrix<float64> r(m.getShape(), 0.);
            auto iter_m = m.getData();
            for (auto &i: r) {
                if (*iter_m >= 0 || *iter_m <= n)
                    i = nchoosek<float64>(n, *iter_m) * pow(p, *iter_m) * pow(1 - p, n - *iter_m);
                iter_m++;
            }
            return r;
        }

        /**
         * Map every element of the matrix through the binomial distribution
         * @param m must be a real-valued matrix; floating-point values are rounded
         * @param n matrix the number of trials of the binomial distribution
         * @param p matrix the success probability of the binomial distribution
         * @return floating-point matrix: the matrix elements mapped through the binomial distribution
         */
        template<RealType RT>
        auto binopdf(const Matrix<RT> &m, Matrix<float64> n, Matrix<float64> p) -> Matrix<float64> {
            n = n.broadcast(m.getShape());
            p = p.broadcast(m.getShape());

            Matrix<float64> r(m.getShape(), 0.);
            auto iter_m = m.getData();
            auto iter_n = n.getData();
            auto iter_p = p.getData();
            for (auto &i: r) {
                if (*iter_m >= 0 || *iter_m <= n)
                    i = nchoosek<float64>(*iter_n, *iter_m) * pow(*iter_p, *iter_m) *
                        pow(1 - *iter_p, *iter_n - *iter_m);
                iter_m++;
                iter_n++;
                iter_p++;
            }
            return r;
        }

        /**
         * Generate a random matrix from the binomial distribution
         * @param n the number of trials of the binomial distribution
         * @param p the success probability of the binomial distribution
         * @param shape the shape of the target matrix
         * @return floating-point matrix: a random matrix from the binomial distribution
         */
        auto binornd(float64 n = 1, float64 p = 0.5, const mshape_type &shape = {1}) -> Matrix<real_max> {
            if (std::round(n) != n || p < 0 || p > 1)
                return Matrix<real_max>(shape, NAN);
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::binomial_distribution<> bino(static_cast<int>(n), p);
            for (auto &i: r)
                i = bino(gen);
            return r;
        }

        /**
         * Generate a random matrix from multiple binomial distributions
         * @param n the number of trials of the binomial distribution
         * @param p the success probability of the binomial distribution
         * @return floating-point matrix: a random matrix from the binomial distribution
         */
        auto binornd(const Matrix<float_max> &n, const Matrix<float_max> &p) -> Matrix<real_max> {
            if (n.getShape() != p.getShape())
                throw_shape_exception("n matrix and p matrix have different sizes");
            Matrix<real_max> r(n.getShape(), 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            for (auto iter = r.begin(), iter_n = n.begin(), iter_p = p.begin();
                 iter != r.end(); ++iter, ++iter_n, ++iter_p) {
                if (std::round(*iter_n) != *iter_n || *iter_p < 0 || *iter_p > 1)
                    *iter = NAN;
                else {
                    std::binomial_distribution<> bino(static_cast<int>(*iter_n), *iter_p);
                    *iter = bino(gen);
                }
            }
            return r;
        }

        /**
         * Generate a random matrix from multiple binomial distributions
         * @param n the number of trials of the binomial distribution
         * @param p the success probability of the binomial distribution
         * @param shape the size of the target matrix
         * @return floating-point matrix: a random matrix from the binomial distribution
         */
        auto binornd(const Matrix<float_max> &n, const Matrix<float_max> &p,
                     const mshape_type &shape = {1}) -> Matrix<real_max> {
            if (n.getShape() != shape || p.getShape() != shape)
                throw_shape_exception("n matrix or p matrix differs in shape from the specified matrix");
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            for (auto iter = r.begin(), iter_n = n.begin(), iter_p = p.begin();
                 iter != r.end(); ++iter, ++iter_n, ++iter_p) {
                if (std::round(*iter_n) != *iter_n || *iter_p < 0 || *iter_p > 1)
                    *iter = NAN;
                else {
                    std::binomial_distribution<> bino(static_cast<int>(*iter_n), *iter_p);
                    *iter = bino(gen);
                }
            }
            return r;
        }

        /**
         * The mean and variance of the binomial distribution
         * @return floating-point matrix tuple {n*p,n*p*(1-p)}
         */
        template<RealType RT1, RealType RT2>
        auto binostat(RT1 n, RT2 p) -> tuple<real_max, real_max> {
            //check n/p legal
            if ((isFloat<RT1>() && std::round(n) != n) || p < 0 || p > 1)
                return {NAN, NAN};
            auto mu = n * p;
            return {mu, mu * (1 - p)};
        }

        /**
         * The mean and variance of the binomial distribution
         * @return floating-point matrix tuple {mu,sigma*sigma}
         */
        template<RealType RT1, RealType RT2>
        auto binostat(const Matrix<RT1> &n, const Matrix<RT2> &p) -> tuple<Matrix<real_max>, Matrix<real_max>> {
            auto mu = mul(n, p);
            return {mu, mul(mu, 1 - p)};
        }
    }
    namespace Math {
        //Continuous uniform distribution
        /**
         * Compute the value of the continuous uniform distribution function
         * @param x the x value passed to the function; x must lie within [a,b]
         * @param a the left boundary of the continuous uniform function; defaults to 0 for the standard distribution
         * @param b the right boundary of the continuous uniform function; defaults to 1 for the standard distribution
         */
        auto unifpdf(float64 x, float64 a = 0, float64 b = 1) -> float64 {
            if (x < a || x > b)
                return 0;
            return 1 / (b - a);
        }

        /**
         * Compute the value of the continuous uniform distribution function
         * @param x the x value passed to the function; x must lie within [a,b]
         * @param a the left boundary of the continuous uniform function; defaults to 0 for the standard distribution
         * @param b the right boundary of the continuous uniform function; defaults to 1 for the standard distribution
         */
        auto unifpdf(const Matrix<float64> &x, const Matrix<float64> &a = {0},
                     const Matrix<float64> &b = {1}) -> Matrix<float64> {
            bool x_single = x.isSingle();
            bool a_single = a.isSingle();
            bool b_single = b.isSingle();
            if (a.getShape() != b.getShape() || (x.getShape() != a.getShape() && !a_single))
                throw_shape_exception("unifpdf: matrices a,b must have the same size, and must equal the size of matrix x");
            mshape_type shape;
            if (!x.isSingle())
                shape = x.getShape();
            else
                shape = a.getShape();
            Matrix<float64> r(shape, 0.);
            auto iter_x = x.begin();
            auto iter_a = a.begin();
            auto iter_b = b.begin();
            for (auto &i: r) {
                i = unifpdf(*iter_x, *iter_a, *iter_b);
                if (!x_single)
                    iter_x++;
                if (!a_single)
                    iter_a++;
                if (!b_single)
                    iter_b++;
            }
            return r;
        }

        /**
        * Compute the expected value and variance of the continuous uniform distribution
        * @param a the left boundary of the continuous uniform function; defaults to 0 for the standard distribution
        * @param b the right boundary of the continuous uniform function; defaults to 1 for the standard distribution
        */
        auto unifstat(float64 a, float64 b) -> tuple<float64, float64> {
            return {(a + b) / 2, pow((a - b), 2) / 12};
        }

        /**
        * Compute the expected value and variance of the continuous uniform distribution
        * @param a the left boundary of the continuous uniform function; defaults to 0 for the standard distribution
        * @param b the right boundary of the continuous uniform function; defaults to 1 for the standard distribution
        */
        auto unifstat(const Matrix<float64> &a, const Matrix<float64> &b) -> tuple<Matrix<float64>, Matrix<float64>> {
            if (a.getShape() != b.getShape())
                throw_shape_exception("unifstat: matrices a b must have the same size");
            return {(a + b) / 2, ((a - b) ^ 2) / 12};
        }
    }

    namespace Math {
        /**
         * Generate a random matrix from the (0,1) uniform distribution
         * @param shape the dimensions of the target matrix
         * @return floating-point matrix: a random matrix from the (0,1) uniform distribution
         */
        auto rand(const mshape_type &shape) -> Matrix<real_max> {
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
        auto rand(msize_type n) -> Matrix<real_max> {
            return rand({n, n});
        }

        /**
         * Generate a matrix from the [imin,imax] uniform distribution
         * @param imin the minimum value of the uniform distribution
         * @param imax the maximum value of the uniform distribution
         * @param shape the shape of the target matrix
         * @return integer matrix: a random matrix from the [imin,imax] uniform distribution
         */
        auto randi(int64 imin, int64 imax, const mshape_type &shape) -> Matrix<int64> {
            if (imin > imax)
                throw_value_error("randi requires the minimum of the integer uniform distribution to be less than or equal to the maximum");
            Matrix<int64> r(shape, 0.);
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
        auto randi(int64 imax, const mshape_type &shape) -> Matrix<int64> {
            return randi(1, imax, shape);
        }

        /**
         * Generate a matrix from the [imin,imax] uniform distribution
         * @param imin the minimum value of the uniform distribution
         * @param imax the maximum value of the uniform distribution
         * @param n indicates that an n*n matrix is generated
         * @return integer matrix: a random matrix from the [imin,imax] uniform distribution
         */
        auto randi(int64 imin, int64 imax, msize_type n) -> Matrix<int64> {
            return randi(imin, imax, {n, n});
        }

        /**
         * Generate a matrix from the [1,imax] uniform distribution
         * @param imax the maximum value of the uniform distribution
         * @param n indicates that an n*n matrix is generated
         * @return integer matrix: a matrix from the [1,imax] uniform distribution
         */
        auto randi(int64 imax, msize_type n) -> Matrix<int64> {
            return randi(1, imax, {n, n});
        }

        /**
         * Generate a standard normal distribution matrix
         * @param shape the shape of the target matrix
         * @return floating-point matrix: a standard normal distribution matrix
         */
        auto randn(const mshape_type &shape) -> Matrix<real_max> {
            return rand(shape);
        }

        /**
         * Generate a standard normal distribution matrix
         * @param n indicates that an n*n matrix is generated
         * @return floating-point matrix: a standard normal distribution matrix
         */
        auto randn(int64 n) -> Matrix<real_max> {
            return randn({n, n});
        }

        template<Calculable T>
        class val_idx {
        public:
            T val = 0;
            size_t idx = 0;

            val_idx() = default;

            val_idx(T val, size_t idx) : val(val), idx(idx) {}

            ~val_idx() = default;
        };

        /**
         * Generate a random permutation of integers
         * @param n the number of integers in the sampling interval
         * @return integer row array
         * @hint uses an offset-based algorithm
         */
        template<IntType IT>
        auto randperm(IT n) -> Matrix<IT> {
            if (n < 0) throw_value_error("randperm: n should be positive");
            unique_ptr<val_idx<real_max>> val_idx_seq(new val_idx<real_max>[n]);
            auto rand_seq = rand(mshape_type{n});
            for (IT i = 0; i < n; ++i) {
                val_idx_seq.get()[i].val = rand_seq(i);
                val_idx_seq.get()[i].idx = i + 1;
            }
            std::sort(&val_idx_seq.get()[0], &val_idx_seq.get()[n],
                      [](val_idx<real_max> v1, val_idx<real_max> v2) { return v1.val < v2.val; });
            Matrix<IT> r({n}, 0.);
            for (IT i = 0; i < n; ++i)
                r(i) = val_idx_seq.get()[i].idx;
            return r;
        }

        /**
         * Generate a random permutation of integers
         * @param n the number of integers in the sampling interval
         * @param k the number of integers to select
         * @return integer row array
         */
        template<IntType IT>
        auto randperm(IT n, IT k) -> Matrix<IT> {
            if (n < 0) throw_value_error("randperm: n should be positive");
            if (k < 0 || k > n) throw_value_error("randperm: k value should be greater than or equal to 0 or less than or equal to n");
            unique_ptr<val_idx<real_max>> val_idx_seq(new val_idx<real_max>[n]);
            auto rand_seq = rand(mshape_type{n});
            for (IT i = 0; i < n; ++i) {
                val_idx_seq.get()[i].val = rand_seq(i);
                val_idx_seq.get()[i].idx = i + 1;
            }
            std::sort(&val_idx_seq.get()[0], &val_idx_seq.get()[n],
                      [](val_idx<real_max> v1, val_idx<real_max> v2) { return v1.val < v2.val; });
            Matrix<IT> r({k}, 0.);
            for (IT i = 0; i < k; ++i)
                r(i) = val_idx_seq.get()[i].idx;
            return r;
        }
    }

    namespace Math {
//        //Chi-squared distribution
//        /**
//         * Compute the value of the chi-squared distribution function
//         * @param x
//         * @param nu
//         */
//        auto chi2cdf(float64 x, float64 nu) {
//            std::chi_squared_distribution<float64> chi2(nu);
//
//        }
        /**
         * Compute the value of the chi-squared distribution function
         * @param x the x value passed to the function
         * @param nu the degrees of freedom of the chi-squared distribution
         * @hint std::tgamma function
         */
        auto chi2pdf(float64 x, float64 nu) -> float64 {
            if (x < 0) return 0;
            if (nu < 0) return NAN;
            auto v2 = nu / 2;
            return std::pow(x, v2 - 1) * std::exp(-x / 2) / std::exp2(v2) / std::tgamma(v2);
        }

        /**
         * Compute the value of the chi-squared distribution function
         * @param x the x value passed to the function
         * @param nu the degrees of freedom of the chi-squared distribution
         * @hint the algorithm can be optimized using matlab's find-based exclusion method
         */
        auto chi2pdf(const Matrix<float64> &x, const Matrix<float64> &nu) -> Matrix<float64> {
            if (x.getShape() != nu.getShape() && !x.isSingle() && !nu.isSingle())
                throw_shape_exception("chi2pdf: x and nu have different dimensions");
            mshape_type shape = nu.isSingle() ? x.getShape() : nu.getShape();
            Matrix<float64> r(shape, 0.);
            auto iter_x = x.begin();
            auto iter_nu = nu.begin();
            for (auto &i: r) {
                i = chi2pdf(*iter_x, *iter_nu);
                if (!x.isSingle())
                    iter_x++;
                if (!nu.isSingle())
                    iter_nu++;
            }
            return r;
        }
    }
    namespace Math {
        //Exponential distribution
        /**
         * Compute the value of the exponential distribution function
         * @param x the x value passed to the function
         * @param mu the mean of the exponential distribution
         */
        auto exppdf(float64 x, float64 mu = 1) -> float64 {
            return 1 / mu * std::exp(-x / mu);
        }

        /**
         * Compute the value of the exponential distribution function
         * @param x the x value passed to the function
         * @param mu the mean of the exponential distribution
         */
        auto exppdf(const Matrix<float64> &x, const Matrix<float64> &mu = {1}) -> Matrix<float64> {
            return 1 / mu * exp(-x / mu);
        }

        /**
         * Generate a random matrix from the exponential distribution
         * @param mu the mean of the exponential distribution
         * @param shape the size of the target matrix
         */
        auto exprnd(float64 mu, const mshape_type &shape = {1}) -> Matrix<float64> {
            Matrix<float64> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::exponential_distribution<float64> exp_dis(mu);
            for (auto &i: r)
                i = exp_dis(gen);
            return r;
        }

        /**
         * The mean and variance of the exponential distribution
         * @return floating-point matrix tuple {mu,mu*mu}
         */
        template<RealType RT>
        auto expstat(const RT &mu) -> tuple<real_max, real_max> {
            real_max tmp = mu;
            return {tmp, tmp * tmp};
        }

        /**
         * The mean and variance of the exponential distribution
         * @return floating-point matrix tuple {mu,mu*mu}
         */
        template<RealType RT>
        auto expstat(const Matrix<RT> &mu) -> tuple<Matrix<real_max>, Matrix<real_max>> {
            Matrix<real_max> tmp(mu);
            return {tmp, tmp ^ 2};
        }
    }

    namespace Math {
        //Poisson distribution
        /**
         * Compute the value of the Poisson distribution function
         * @param x the x value passed to the function; must be an integer; non-integer values return 0
         * @param lambda the rate parameter of the Poisson distribution, must be positive
         */
        auto poisspdf(float_max x, float_max lambda) -> float_max {
            if (x < 0 || x != std::round(x))
                return 0;
            if (lambda < 0)
                return NAN;
            return pow(lambda, x) / matsar::factorial<float_max>(x) * std::exp(-lambda);
        }

        /**
         * Compute the value of the Poisson distribution function
         * @param x the x value passed to the function; must be an integer; non-integer values return 0
         * @param lambda the rate parameter of the Poisson distribution, must be positive
         */
        auto poisspdf(const Matrix<float64> &x, const Matrix<float64> &lambda) -> Matrix<float64> {
            if (x.getShape() != lambda.getShape() && !x.isSingle() && !lambda.isSingle())
                throw_shape_exception("poisspdf: x and lambda have different dimensions");
            mshape_type shape;
            if (!x.isSingle())
                shape = x.getShape();
            else
                shape = lambda.getShape();
            Matrix<float64> r(shape, 0.);
            auto iter_x = x.begin();
            auto iter_lambda = lambda.begin();
            for (auto &i: r) {
                i = poisspdf(*iter_x, *iter_lambda);
                if (!x.isSingle())
                    iter_x++;
                if (!lambda.isSingle())
                    iter_lambda++;
            }
            return r;
        }

        /**
         * Generate a random matrix from the Poisson distribution
         * @param lambda the rate parameter of the Poisson distribution, must be positive
         * @param shape the size of the target matrix
         */
        auto poissrnd(float64 lambda, const mshape_type &shape = {1}) -> Matrix<float64> {
            Matrix<float64> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::poisson_distribution<> poisson(lambda);
            for (auto &i: r)
                i = poisson(gen);
            return r;
        }
    }
    namespace Math {
        //Rayleigh distribution
        /**
         * Compute the value of the Rayleigh distribution function
         * @param x the x value passed to the function; must be non-negative; negative values return 0
         * @param b the scale value of the Rayleigh distribution
         */
        auto raylpdf(float64 x, float64 b) -> float64 {
            if (x < 0) return 0;
            if (b <= 0) return NAN;
            auto b2 = b * b;
            return x / b2 * std::exp(-x * x / 2 / b2);
        }

        /**
         * Compute the value of the Rayleigh distribution function
         * @param x the x value passed to the function; must be non-negative; negative values return 0
         * @param b the scale value of the Rayleigh distribution
         */
        auto raylpdf(const Matrix<float64> &x, const Matrix<float64> &b) -> Matrix<float64> {
            bool x_single = x.isSingle();
            bool b_single = b.isSingle();
            if (x.getShape() != b.getShape() && !x_single && !b_single)
                throw_shape_exception("raylpdf: x and b have different sizes");
            mshape_type shape;
            if (!x_single)
                shape = x.getShape();
            else
                shape = b.getShape();
            Matrix<float64> r(shape, 0.);
            auto iter_x = x.begin();
            auto iter_b = b.begin();
            for (auto &i: r) {
                i = raylpdf(*iter_x, *iter_b);
                if (!x_single)
                    iter_x++;
                if (!b_single)
                    iter_b++;
            }
            return r;
        }
    }
    namespace Math {
        //Weibull distribution
        /**
         * Compute the value of the Weibull distribution function
         * @param x the x value passed to the function; must be non-negative; negative values give a Weibull value of 0
         * @param lambda the scale parameter of the Weibull distribution
         * @param k the shape parameter of the Weibull distribution
         */
        auto weibpdf(float64 x, float64 lambda, float64 k) -> float64 {
            if (x < 0) return 0;
            if (lambda <= 0 || k <= 0) return NAN;
            auto lambda_re = 1 / lambda;
            auto x_lambda_re = x * lambda_re;
            auto stat1 = std::pow(x_lambda_re, k - 1);
            return k * lambda_re * stat1 * std::exp(-stat1 * x_lambda_re);
        }

        /**
         * Compute the value of the Weibull distribution function
         * @param x the x value passed to the function; must be non-negative; negative values give a Weibull value of 0
         * @param lambda the scale parameter of the Weibull distribution
         * @param k the shape parameter of the Weibull distribution
         */
        auto weibpdf(const Matrix<float64> &x, const Matrix<float64> &lambda, const Matrix<float64> &k) -> Matrix<float64> {
            bool x_single = x.isSingle();
            bool lambda_single = lambda.isSingle();
            bool k_single = k.isSingle();
            if (lambda.getShape() != k.getShape() || (!lambda_single && x.getShape() != lambda.getShape()))
                throw_shape_exception("weibpdf: the lambda k matrices must have the same size, and match the size of x");
            mshape_type shape;
            if (!x_single)
                shape = x.getShape();
            else
                shape = lambda.getShape();
            Matrix<float64> r(shape, 0.);
            auto iter_x = x.begin();
            auto iter_lambda = lambda.begin();
            auto iter_k = k.begin();
            for (auto &i: r) {
                i = weibpdf(*iter_x, *iter_lambda, *iter_k);
                if (!x_single)
                    iter_x++;
                if (!lambda_single)
                    iter_lambda++;
                if (!k_single)
                    iter_k++;
            }
            return r;
        }
    }
}

#endif //MATSAR_PROBABILITY_HPP
