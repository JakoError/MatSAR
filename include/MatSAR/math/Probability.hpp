//
// Created by JakoError on 2022/7/11.
//

#ifndef MATSAR_PROBABILITY_HPP
#define MATSAR_PROBABILITY_HPP

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

#include <random>
#include <algorithm>

namespace matsar {
///矩阵概率计算函数
    namespace Math {
        /**
         * 返回矩阵对应维度最大值
         * @param m 支持所有Calculable矩阵
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         * @return 矩阵对应维度最大值
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
         * 返回矩阵对应维度最小值
         * @param m 支持所有Calculable矩阵
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         * @return 矩阵对应维度最小值
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
         * @related range函数等价于调用min和max函数
         */
        template<Calculable dType>
        auto range(const Matrix<dType> &m, msize_type dim = 1) -> tuple<Matrix<dType>, Matrix<dType>> {
            return {min(m, dim), max(m, dim)};
        }

        /**
         * 返回矩阵对应维度中位数
         * @param m 支持所有Calculable矩阵
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         * @return 矩阵对应维度中位数
         */
        template<Calculable TargetType, Calculable dType>
        auto median(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<TargetType> {
            auto dim_idx = dim - 1;
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0) {
                auto sort_m = m.sort();
                auto iter = sort_m.getData() + sort_m.getSize() / 2;
                if (sort_m.getSize() % 2 == 0) {
                    return {(*(iter - 1) + *iter) / 2.};
                } else {
                    return {*iter / 1.};
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
            Matrix<TargetType> r(r_shape, 0., m.getData());

            auto iter_r = r.getData();
            auto iter_m = sort_m.getData();
            if (shape[dim_idx] % 2 == 0) {
                auto pick = (step_num / 2 - 1) * step_size;
                iter_m += pick;
                for (msize_type i = 0; i < unit_num; ++i) {
                    for (msize_type j = 0; j < step_size; ++j) {
                        *(iter_r++) = (*(iter_m + j) + *(iter_m + j + step_size)) / 2.;
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

        template<IntType IT>
        auto median(const Matrix<IT> &m, msize_type dim = 1) -> Matrix<real_max> {
            return median<real_max, IT>(m, dim);
        }

        template<FloatType FT>
        auto median(const Matrix<FT> &m, msize_type dim = 1) -> Matrix<FT> {
            return median<FT, FT>(m, dim);
        }

        template<ComplexType CT>
        auto median(const Matrix<CT> &m, msize_type dim = 1) -> Matrix<CT> {
            return median<CT, CT>(m, dim);
        }

        /**
         * mean求n个数的平均值
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<Calculable TargetType, Calculable dType>
        auto mean(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<TargetType> {
            auto dim_idx = dim - 1;
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0)
                return sum<TargetType, dType>(m / float_max(m.getSize()), dim);
            return sum<TargetType, dType>(m / float_max(m.getShape()[dim_idx]), dim);
        }

        template<RealType RT>
        auto mean(const Matrix<RT> &m, msize_type dim = 1) -> Matrix<real_max> {
            return mean<real_max, RT>(m, dim);
        }

        template<ComplexType CT>
        auto mean(const Matrix<CT> &m, msize_type dim = 1) -> Matrix<real_max> {
            return mean<real_max, CT>(m, dim);
        }
    }
    namespace Math {
        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<Calculable TargetType, Calculable dType>
        auto var(const Matrix<dType> &m, bool w = false, msize_type dim = 1) -> Matrix<TargetType> {
            msize_type dim_idx = dim - 1;
            if (m.isEmpty()) return {};
            if (dim_idx >= m.getDims()) return {m};
            auto n = m.getSize();
            auto denom = w ? n - 1 : n;
            return {sum<TargetType, dType>(abs(m - sum<TargetType, dType>(m, dim) / n) ^ 2, dim) / denom};
        }

        template<RealType RT>
        auto var(const Matrix<RT> &m, bool w = false, msize_type dim = 1) -> Matrix<real_max> {
            return var<real_max, RT>(m, w, dim);
        }

        template<ComplexType CT>
        auto var(const Matrix<CT> &m, bool w = false, msize_type dim = 1) -> Matrix<complex_max> {
            return var<complex_max, CT>(m, w, dim);
        }
    }
    ///分布函数
    namespace Math {
//        static const float_max PI = static_cast<float_max>(3.14159265358979323846);

        /**
         * 将矩阵内的所有元素经过正态分布映射
         * @param m 要求为实数矩阵
         * @param mu 正态分布的均值
         * @param sigma 正态分布的方差
         * @return 浮点矩阵：矩阵元素正态分布映射
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
         * 将矩阵内的所有元素经过正态分布映射
         * @param m 要求矩阵为实数矩阵
         * @param mu矩阵 正态分布的均值，会对mu矩阵进行broadcast操作
         * @param sigma矩阵 正态分布的方差，会对sigma矩阵进行broadcast操作
         * @return 浮点矩阵：矩阵元素正态分布映射
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
         * 生成正态分布的随机矩阵
         * @param mu 正态分布的均值
         * @param sigma 正态分布的方差
         * @param shape 目标矩阵的shape
         * @return 浮点矩阵：正态分布的随机矩阵
         */
        auto normrnd(float_max mu = 0., float_max sigma = 1., const mshape_type &shape = {1}) -> Matrix<real_max> {
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            normal_distribution <float_max> normal(mu, sigma);
            for (auto &i: r)
                i = normal(gen);
            return r;
        }

        /**
         * 生成正态分布的随机矩阵
         * @param mu矩阵 正态分布的均值，会对mu矩阵进行broadcast操作
         * @param sigma矩阵 正态分布的方差，会对sigma矩阵进行broadcast操作
         * @param shape 目标矩阵的shape
         * @return 浮点矩阵：正态分布的随机矩阵
         */
        auto normrnd(const Matrix<float_max> &mu, const Matrix<float_max> &sigma,
                     const mshape_type &shape = {1}) -> Matrix<real_max> {
            if (mu.getShape() != shape || sigma.getShape() != shape)
                throw_shape_exception("mu矩阵或sigma矩阵大小与指定shape不同");
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            for (auto iter = r.begin(), iter_mu = mu.begin(), iter_sigma = sigma.begin();
                 iter != r.end(); ++iter, ++iter_mu, ++iter_sigma) {
                normal_distribution <float_max> normal(*iter_mu, *iter_sigma);
                *iter = normal(gen);
            }
            return r;
        }

        /**
         * 正态分布的均值和方差
         * @return 浮点矩阵tuple {mu,sigma*sigma}
         */
        template<RealType RT1, RealType RT2>
        auto normstat(RT1 mu, RT2 sigma) -> tuple<real_max, real_max> {
            return {mu, mul(sigma, sigma)};
        }

        /**
         * 正态分布的均值和方差
         * @return 浮点矩阵tuple {mu,sigma*sigma}
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
         * 将矩阵内的所有元素经过二项分布映射
         * @param m 要求为实数矩阵，若浮点四舍五入
         * @param n 二项分布的实验次数
         * @param p 二项分布的发生概率
         * @return 浮点矩阵：矩阵元素二项分布映射
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
         * 将矩阵内的所有元素经过二项分布映射
         * @param m 要求为实数矩阵，若浮点四舍五入
         * @param n matrix 二项分布的实验次数
         * @param p matrix 二项分布的发生概率
         * @return 浮点矩阵：矩阵元素二项分布映射
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
         * 生成二项分布的随机矩阵
         * @param mu 二项分布的均值
         * @param sigma 二项分布的方差
         * @param shape 目标矩阵的shape
         * @return 浮点矩阵：二项分布的随机矩阵
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
         * 生成多个二项分布的随机矩阵
         * @param mu矩阵 二项分布的均值
         * @param sigma矩阵 二项分布的方差
         * @return 浮点矩阵：二项分布的随机矩阵
         */
        auto binornd(const Matrix<float_max> &n, const Matrix<float_max> &p) -> Matrix<real_max> {
            if (n.getShape() != p.getShape())
                throw_shape_exception("n矩阵和p矩阵大小不同");
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
         * 生成多个二项分布的随机矩阵
         * @param mu矩阵 二项分布的均值
         * @param sigma矩阵 二项分布的方差
         * @param shape 目标矩阵的大小
         * @return 浮点矩阵：二项分布的随机矩阵
         */
        auto binornd(const Matrix<float_max> &n, const Matrix<float_max> &p,
                     const mshape_type &shape = {1}) -> Matrix<real_max> {
            if (n.getShape() != shape || p.getShape() != shape)
                throw_shape_exception("n矩阵或p矩阵与指定矩阵形状大小不同");
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
         * 二项分布的均值和方差
         * @return 浮点矩阵tuple {n*p,n*p*(1-p)}
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
         * 二项分布的均值和方差
         * @return 浮点矩阵tuple {mu,sigma*sigma}
         */
        template<RealType RT1, RealType RT2>
        auto binostat(const Matrix<RT1> &n, const Matrix<RT2> &p) -> tuple<Matrix<real_max>, Matrix<real_max>> {
            auto mu = mul(n, p);
            return {mu, mul(mu, 1 - p)};
        }
    }
    namespace Math {
        //连续均匀分布
        /**
         * 求连续均匀分布函数值
         * @param x 函数传入x值，x值需在[a,b]内
         * @param a 连续均匀函数左边界值，默认为标准分布0
         * @param b 连续均匀函数右边界值，默认为标准分布1
         */
        auto unifpdf(float64 x, float64 a = 0, float64 b = 1) -> float64 {
            if (x < a || x > b)
                return 0;
            return 1 / (b - a);
        }

        /**
         * 求连续均匀分布函数值
         * @param x 函数传入x值，x值需在[a,b]内
         * @param a 连续均匀函数左边界值，默认为标准分布0
         * @param b 连续均匀函数右边界值，默认为标准分布1
         */
        auto unifpdf(const Matrix<float64> &x, const Matrix<float64> &a = {0},
                     const Matrix<float64> &b = {1}) -> Matrix<float64> {
            bool x_single = x.isSingle();
            bool a_single = a.isSingle();
            bool b_single = b.isSingle();
            if (a.getShape() != b.getShape() || (x.getShape() != a.getShape() && !a_single))
                throw_shape_exception("unifpdf: a,b两矩阵大小应该相同，并且等于x矩阵大小");
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
        * 求连续均匀分布期望与方差
        * @param a 连续均匀函数左边界值，默认为标准分布0
        * @param b 连续均匀函数右边界值，默认为标准分布1
        */
        auto unifstat(float64 a, float64 b) -> tuple<float64, float64> {
            return {(a + b) / 2, pow((a - b), 2) / 12};
        }

        /**
        * 求连续均匀分布期望与方差
        * @param a 连续均匀函数左边界值，默认为标准分布0
        * @param b 连续均匀函数右边界值，默认为标准分布1
        */
        auto unifstat(const Matrix<float64> &a, const Matrix<float64> &b) -> tuple<Matrix<float64>, Matrix<float64>> {
            if (a.getShape() != b.getShape())
                throw_shape_exception("unifstat: a b两矩阵应该有相同的大小");
            return {(a + b) / 2, ((a - b) ^ 2) / 12};
        }
    }

    namespace Math {
        /**
         * 生成(0,1)均匀分布的随机矩阵
         * @param shape 目标矩阵维度
         * @return 浮点矩阵：(0,1)均匀分布的随机矩阵
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
         * 生成(0，1)均匀分布矩阵
         * @param n 表示生成n*n的矩阵
         * @return 浮点矩阵：(0,1)均匀分布的随机矩阵
         */
        auto rand(msize_type n) -> Matrix<real_max> {
            return rand({n, n});
        }

        /**
         * 生成[imin，imax]均匀分布矩阵
         * @param imin 表示均匀分布最小值
         * @param imax 表示均匀分布最大值
         * @param shape 目标矩阵shape
         * @return 整数矩阵：[imin，imax]均匀分布的随机矩阵
         */
        auto randi(int64 imin, int64 imax, const mshape_type &shape) -> Matrix<int64> {
            if (imin > imax)
                throw_value_error("randi中要求整数均匀分布最小值应该小于等于最大值");
            Matrix<int64> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int64> uniform(imin, imax);
            for (auto &i: r)
                i = uniform(gen);
            return r;
        }

        /**
         * 生成[1，imax]均匀分布矩阵
         * @param imax 表示均匀分布最小值
         * @param shape 目标矩阵shape
         * @return 整数矩阵：[1，imax]均匀分布矩阵
         */
        auto randi(int64 imax, const mshape_type &shape) -> Matrix<int64> {
            return randi(1, imax, shape);
        }

        /**
         * 生成[imin，imax]均匀分布矩阵
         * @param imin 表示均匀分布最小值
         * @param imax 表示均匀分布最大值
         * @param n 表示生成n*n的矩阵
         * @return 整数矩阵：[imin，imax]均匀分布的随机矩阵
         */
        auto randi(int64 imin, int64 imax, msize_type n) -> Matrix<int64> {
            return randi(imin, imax, {n, n});
        }

        /**
         * 生成[1，imax]均匀分布矩阵
         * @param imax 表示均匀分布最大值
         * @param n 表示生成n*n的矩阵
         * @return 整数矩阵：[1，imax]均匀分布矩阵
         */
        auto randi(int64 imax, msize_type n) -> Matrix<int64> {
            return randi(1, imax, {n, n});
        }

        /**
         * 生成标准正态分布矩阵
         * @param shape 目标矩阵shape
         * @return 浮点矩阵：标准正态分布矩阵
         */
        auto randn(const mshape_type &shape) -> Matrix<real_max> {
            return rand(shape);
        }

        /**
         * 生成标准正态分布矩阵
         * @param n 表示生成n*n的矩阵
         * @return 浮点矩阵：标准正态分布矩阵
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
         * 生成整数的随机排列
         * @param n 采样间隔中的整数数量
         * @return 整数行数组
         * @hint 使用偏移量算法
         */
        template<IntType IT>
        auto randperm(IT n) -> Matrix<IT> {
            if (n < 0) throw_value_error("randperm: n 应该为正数");
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
         * 生成整数的随机排列
         * @param n 采样间隔中的整数数量
         * @param k 所选整数的数量
         * @return 整数行数组
         */
        template<IntType IT>
        auto randperm(IT n, IT k) -> Matrix<IT> {
            if (n < 0) throw_value_error("randperm: n 应该为正数");
            if (k < 0 || k > n) throw_value_error("randperm: k值应该大于等于0或小于等于n");
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
//        //卡方分布
//        /**
//         * 求卡方分布函数值
//         * @param x
//         * @param nu
//         */
//        auto chi2cdf(float64 x, float64 nu) {
//            std::chi_squared_distribution<float64> chi2(nu);
//
//        }
        /**
         * 求卡方分布函数值
         * @param x 函数传入x值
         * @param nu 卡方分布自由度
         * @hint std::tgamma function
         */
        auto chi2pdf(float64 x, float64 nu) -> float64 {
            if (x < 0) return 0;
            if (nu < 0) return NAN;
            auto v2 = nu / 2;
            return std::pow(x, v2 - 1) * std::exp(-x / 2) / std::exp2(v2) / std::tgamma(v2);
        }

        /**
         * 求卡方分布函数值
         * @param x 函数传入x值
         * @param nu 卡方分布自由度
         * @hint 可使用matlab中find排除法优化算法
         */
        auto chi2pdf(const Matrix<float64> &x, const Matrix<float64> &nu) -> Matrix<float64> {
            if (x.getShape() != nu.getShape() && !x.isSingle() && !nu.isSingle())
                throw_shape_exception("poisspdf: x 与 nu 维度不同");
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
        //指数分布
        /**
         * 求指数分布函数值
         * @param x 函数传入x值
         * @param mu 指数分布平均值
         */
        auto exppdf(float64 x, float64 mu = 1) -> float64 {
            return 1 / mu * std::exp(-x / mu);
        }

        /**
         * 求指数分布函数值
         * @param x 函数传入x值
         * @param mu 指数分布平均值
         */
        auto exppdf(const Matrix<float64> &x, const Matrix<float64> &mu = {1}) -> Matrix<float64> {
            return 1 / mu * exp(-x / mu);
        }

        /**
         * 生成指数分布随机矩阵
         * @param mu 指数分布平均值
         * @param shape 目标矩阵的大小
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
         * 指数分布的均值和方差
         * @return 浮点矩阵tuple {mu,mu*mu}
         */
        template<RealType RT>
        auto expstat(const RT &mu) -> tuple<real_max, real_max> {
            real_max tmp = mu;
            return {tmp, tmp * tmp};
        }

        /**
         * 指数分布的均值和方差
         * @return 浮点矩阵tuple {mu,mu*mu}
         */
        template<RealType RT>
        auto expstat(const Matrix<RT> &mu) -> tuple<Matrix<real_max>, Matrix<real_max>> {
            Matrix<real_max> tmp(mu);
            return {tmp, tmp ^ 2};
        }
    }

    namespace Math {
        //泊松分布
        /**
         * 求泊松分布函数值
         * @param x 函数传入x值，需要整数值，非整数值泊松分布为0
         * @param lambda 指数分布lambda值，速率参数，为正数
         */
        auto poisspdf(float64 x, float64 lambda) -> float64 {
            if (x < 0 || x != std::round(x))
                return 0;
            if (lambda < 0)
                return NAN;
            return pow(lambda, x) / factorial<float64>(x) * std::exp(-lambda);
        }

        /**
         * 求泊松分布函数值
         * @param x 函数传入x值，需要整数值，非整数值泊松分布为0
         * @param lambda 指数分布lambda值，速率参数，为正数
         */
        auto poisspdf(const Matrix<float64> &x, const Matrix<float64> &lambda) -> Matrix<float64> {
            if (x.getShape() != lambda.getShape() && !x.isSingle() && !lambda.isSingle())
                throw_shape_exception("poisspdf: x 与 lambda 维度不同");
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
         * 生成泊松分布随机矩阵
         * @param lambda 指数分布lambda值，速率参数，为正数
         * @param shape 目标矩阵的大小
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
        //瑞利分布
        /**
         * 求瑞利分布函数值
         * @param x 函数传入x值，需要非负值，负值泊松分布为0
         * @param b 瑞利分布比例值
         */
        auto raylpdf(float64 x, float64 b) -> float64 {
            if (x < 0) return 0;
            if (b <= 0) return NAN;
            auto b2 = b * b;
            return x / b2 * std::exp(-x * x / 2 / b2);
        }

        /**
         * 求瑞利分布函数值
         * @param x 函数传入x值，需要非负值，负值泊松分布为0
         * @param b 瑞利分布比例值
         */
        auto raylpdf(const Matrix<float64> &x, const Matrix<float64> &b) -> Matrix<float64> {
            bool x_single = x.isSingle();
            bool b_single = b.isSingle();
            if (x.getShape() != b.getShape() && !x_single && !b_single)
                throw_shape_exception("raylpdf: x 与 b 大小不同");
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
        //韦伯分布
        /**
         * 求韦伯分布函数值
         * @param x 函数传入x值，需要非负值，负值韦伯分布为0
         * @param lambda 韦伯分布比例参数
         * @param k 韦伯分布形状参数
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
         * 求韦伯分布函数值
         * @param x 函数传入x值，需要非负值，负值韦伯分布为0
         * @param lambda 韦伯分布比例参数
         * @param k 韦伯分布形状参数
         */
        auto weibpdf(const Matrix<float64> &x, const Matrix<float64> &lambda,const Matrix<float64> &k) -> Matrix<float64> {
            bool x_single = x.isSingle();
            bool lambda_single = lambda.isSingle();
            bool k_single = k.isSingle();
            if (lambda.getShape() != k.getShape() || (!lambda_single && x.getShape() != lambda.getShape()))
                throw_shape_exception("weibpdf: lambda k 两矩阵应该有相同的大小，并且与x大小匹配");
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
                i = weibpdf(*iter_x, *iter_lambda,*iter_k);
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
