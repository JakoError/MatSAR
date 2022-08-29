//
// Created by JakoError on 2022/7/11.
//

#ifndef MATSAR_PROBABILITY_HPP
#define MATSAR_PROBABILITY_HPP

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif //_USE_MATH_DEFINES

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

namespace MatSAR {
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
            auto step_size = MatSAR::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<dType> r(r_shape, 0.,m.getData());

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
            auto step_size = MatSAR::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<dType> r(r_shape, 0.,m.getData());

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
        auto range(const Matrix <dType> &m, msize_type dim = 1) -> tuple<Matrix<dType>, Matrix<dType>> {
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
            auto step_size = MatSAR::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto sort_m = MatSAR::Math::sort(m, dim);
            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<TargetType> r(r_shape, 0.,m.getData());

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
                return sum<TargetType,dType>(m / float_max(m.getSize()), dim);
            return sum<TargetType,dType>(m / float_max(m.getShape()[dim_idx]), dim);
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
            return {sum<TargetType,dType>(abs(m - sum<TargetType,dType>(m, dim) / n) ^ 2, dim) / denom};
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
        static const float_max PI = static_cast<float_max>(3.14159265358979323846);

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
            normal_distribution<float_max> normal(mu, sigma);
            for(auto &i:r)
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
            auto b_mu = mu.broadcast(shape);
            auto b_sigma = sigma.broadcast(shape);
            Matrix<real_max> r(shape, 0.);
            std::random_device rd;
            std::mt19937 gen(rd());
            for(auto iter = r.begin(),iter_mu = b_mu.begin(),iter_sigma = b_sigma.begin();
            iter != r.end();++iter,++iter_mu,++iter_sigma){
                normal_distribution<float_max> normal(*iter_mu, *iter_sigma);
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
            return {mu, sigma * sigma};
        }

        /**
         * 正态分布的均值和方差
         * @return 浮点矩阵tuple {mu,sigma*sigma}
         */
        template<RealType RT1, RealType RT2>
        auto normstat(const Matrix<RT1> &mu, const Matrix<RT2> &sigma) -> tuple<Matrix<real_max>, Matrix<real_max>> {
            Matrix<real_max> b_mu(mu);
            Matrix<real_max> b_sigma(sigma);
            Matrix<real_max>::broadcast(mu,sigma);
            return {b_mu,b_sigma ^ 2};
        }

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
            for(auto &i:r)
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
            for(auto &i:r)
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
    }
}

#endif //MATSAR_PROBABILITY_HPP
