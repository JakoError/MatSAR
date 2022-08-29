//
// Created by JakoError on 2022/7/9.
//

#ifndef MATSAR_LINEARALGEBRA_HPP
#define MATSAR_LINEARALGEBRA_HPP

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

using MatSAR::Util::TypeUtil::eps;

using MatSAR::Util::VectorUtil::Range;

namespace MatSAR {
    namespace Math {
        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<Calculable TargetType, Calculable dType>
        auto sum(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<TargetType> {
            auto dim_idx = dim - 1;
            if (m.isEmpty())
                return {};
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0) {
                TargetType sum = 0.;
                for (const auto &i: m)
                    sum += i;
                return {sum};
            }
            auto shape = m.getShape();
            auto step_size = MatSAR::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<TargetType> r(r_shape, 0.,m.getData());

            auto iter_r = r.getData();
            auto iter_m = m.getData();
            for (msize_type i = 0; i < unit_num; ++i) {
                for (msize_type j = 0; j < step_num; ++j) {
                    for (msize_type k = 0; k < step_size; ++k) {
                        *(iter_r + k) += *(iter_m + k);
                    }
                    iter_m += step_size;
                }
                iter_r += step_size;
            }
            return r;
        }

        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<SIntType SIT>
        auto sum(const Matrix<SIT> &m, msize_type dim = 1) -> Matrix<int_max> {
            return sum<int_max, SIT>(m, dim);
        }

        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<UIntType UIT>
        auto sum(const Matrix<UIT> &m, msize_type dim = 1) -> Matrix<uint_max> {
            return sum<uint_max, UIT>(m, dim);
        }

        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<FloatType FT>
        auto sum(const Matrix<FT> &m, msize_type dim = 1) -> Matrix<float_max> {
            return sum<float_max, FT>(m, dim);
        }

        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<ComplexType CT>
        auto sum(const Matrix<CT> &m, msize_type dim = 1) -> Matrix<complex_max> {
            return sum<complex_max, CT>(m, dim);
        }

    }
    namespace Math {
        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<Calculable TargetType, Calculable dType>
        auto prod(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<TargetType> {
            auto dim_idx = dim - 1;
            if (m.isEmpty())
                return {};
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0) {
                TargetType prod = 1.;
                for (const auto &i: m)
                    prod *= i;
                return {prod};
            }
            auto shape = m.getShape();
            auto step_size = MatSAR::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<TargetType> r(r_shape, 1,m.getData());

            auto iter_r = r.getData();
            auto iter_m = m.getData();
            for (msize_type i = 0; i < unit_num; ++i) {
                for (msize_type j = 0; j < step_num; ++j) {
                    for (msize_type k = 0; k < step_size; ++k) {
                        *(iter_r + k) *= *(iter_m + k);
                    }
                    iter_m += step_size;
                }
                iter_r += step_size;
            }
            return r;
        }

        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<SIntType SIT>
        auto prod(const Matrix<SIT> &m, msize_type dim = 1) -> Matrix<int_max> {
            return prod<int_max, SIT>(m, dim);
        }

        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<UIntType UIT>
        auto prod(const Matrix<UIT> &m, msize_type dim = 1) -> Matrix<uint_max> {
            return prod<uint_max, UIT>(m, dim);
        }

        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<FloatType FT>
        auto prod(const Matrix<FT> &m, msize_type dim = 1) -> Matrix<float_max> {
            return prod<float_max, FT>(m, dim);
        }

        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<ComplexType CT>
        auto prod(const Matrix<CT> &m, msize_type dim = 1) -> Matrix<complex_max> {
            return prod<complex_max, CT>(m, dim);
        }

    }

    namespace Math {
        template<RealType T1, RealType T2>
        auto kron(Matrix<T1> m1, Matrix<T2> m2) -> Matrix<real_max> {
            m1.expandShape(2);
            m2.expandShape(2);
            if (!m1.is2D() || !m2.is2D())
                throw_value_error("kron要求两矩阵均为二维矩阵或可扩展为二维矩阵", true);
            auto shape1 = m1.getShape();
            auto shape2 = m2.getShape();
            auto r_shape = shape1;
            r_shape[0] *= shape2[0];
            r_shape[1] *= shape2[1];
            Matrix<real_max> r(r_shape, 0.,m1.getData());
            for (int64 i = 0; i < shape1[0]; ++i)
                for (int64 j = 0; j < shape1[1]; ++j)
                    for (int64 k = 0; k < shape2[0]; ++k)
                        for (int64 l = 0; l < shape2[1]; ++l)
                            r(i * shape2[0] + k, j * shape2[1] + l) = m1(i, j) * m2(k, l);
            return r;
        }
    }

    namespace Math {
        template<RealType RT>
        auto svd(const Matrix<RT> &A) -> Matrix<float64> {
            Matrix<float64> fA(A);
            fA.expandShape(2);
            if (!fA.is2D())
                throw_value_error("svd需要二维矩阵或可扩展为二维矩阵");
            ///FIXME int64和lapack_int判断转换
            lapack_int m = fA.getShape()[0], n = fA.getShape()[1];
            mshape_type S_shape = {min(m, n)};
            Matrix<float64> S(S_shape, 0.,A.getData());

            unique_ptr<float64> size(new float64);
            LAPACKE_dgesvd_work(CblasColMajor, 'N', 'N', m, n, Matrix<float64>(A).getData(), m, S.getData(),
                                nullptr, 1,
                                nullptr, 1,
                                size.get(), -1);
            lapack_int work_size = (max(3 * min(m, n) + max(m, n), 5 * max(m, n)) + (lapack_int) *size) / 2;
            unique_ptr<float64[]> work(new float64[work_size]);

            auto info = LAPACKE_dgesvd_work(CblasColMajor, 'N', 'N', m, n,
                                            fA.getData(), m, S.getData(),
                                            nullptr, 1, nullptr, 1, work.get(), work_size);
            if (info != 0) {
                A.debug();
                throw_warning("svd works badly error code:" + to_string(info));
            }
            return S;
        }

        auto svd(const Matrix<complex64> &A) -> Matrix<float32> {
            Matrix<complex64> cA(A);
            cA.expandShape(2);
            if (!cA.is2D())
                throw_value_error("svd需要二维矩阵或可扩展为二维矩阵");
            lapack_int m = cA.getShape()[0], n = cA.getShape()[1];
            mshape_type S_shape = {min(m, n)};
            Matrix<float32> S(S_shape, 0.,A.getData());

            unique_ptr<complex64> size(new complex64);
            LAPACKE_cgesvd_work(CblasColMajor, 'N', 'N', m, n, nullptr, m, nullptr,
                                nullptr, 1,
                                nullptr, 1,
                                size.get(), -1, nullptr);

            lapack_int work_size = ((2 * min(m, n) + max(m, n)) + (lapack_int) size->real()) / 2;
            unique_ptr<complex64[]> lwork(new complex64[work_size]);
            unique_ptr<float32[]> rwork(new float32[5 * min(m, n)]);
            LAPACKE_cgesvd_work(CblasColMajor, 'N', 'N', m, n,
                                cA.getData(), m, S.getData(),
                                nullptr, 1, nullptr, 1, lwork.get(), work_size, rwork.get());
            return S;
        }

        auto svd(const Matrix<complex128> &A) -> Matrix<float64> {
            Matrix<complex128> cA(A);
            cA.expandShape(2);
            if (!cA.is2D())
                throw_value_error("svd需要二维矩阵或可扩展为二维矩阵");
            lapack_int m = cA.getShape()[0], n = cA.getShape()[1];
            mshape_type S_shape = {min(m, n)};
            Matrix<float64> S(S_shape, 0.,A.getData());

            unique_ptr<complex128> size(new complex128);
            LAPACKE_zgesvd_work(CblasColMajor, 'N', 'N', m, n, nullptr, m, nullptr,
                                nullptr, 1,
                                nullptr, 1,
                                size.get(), -1, nullptr);

            lapack_int work_size = ((2 * min(m, n) + max(m, n)) + (lapack_int) size->real()) / 2;
            unique_ptr<complex128[]> lwork(new complex128[work_size]);
            unique_ptr<float64[]> rwork(new float64[5 * min(m, n)]);
            LAPACKE_zgesvd_work(CblasColMajor, 'N', 'N', m, n,
                                cA.getData(), m, S.getData(),
                                nullptr, 1, nullptr, 1, lwork.get(), work_size, rwork.get());
            return S;
        }

        template<Calculable dType>
        auto rank(const Matrix<dType> &m, float64 tol) -> msize_type {
            Matrix<float64> s = svd(m);
            return msize_type(sum(s > tol));
        }

        template<Calculable dType>
        auto rank(const Matrix<dType> &m) -> msize_type {
            Matrix<float64> s = svd(m);
            auto tol = m.getLength() * eps(s.max());
            return msize_type(sum(s > tol));
        }
    }

    ///矩阵判断
    namespace Math {
        template<Calculable dType>
        auto all(const Matrix<dType> &m) -> bool {
            if (m.isEmpty())
                return true;
            for (const auto &i: m)
                if (i == 0.)
                    return false;
            return true;
        }

        template<Calculable dType>
        auto any(const Matrix<dType> &m) -> bool {
            if (m.isEmpty())
                return true;
            for (const auto &i: m)
                if (i != 0.)
                    return true;
            return false;
        }
    }
    ///矩阵生成
    namespace Math {
        /**
         * 如果为1维变为二维扩展，如果为二维抽取对角到一维
         */
        template<Calculable dType>
        auto diag(const Matrix<dType> &m) -> Matrix<dType> {
            if (m.isEmpty())
                return {};
            if (m.getDims() > 2)
                throw_value_error("diag函数需要矩阵为2维或1维");
            if (m.isVec()) {
                auto n = m.getShape()[0];
                Matrix<dType> r({n, n}, 0.,m.getData());
                for (msize_type i = 0; i < n; ++i)
                    r(i, i) = m(i);
                return r;
            } else {
                auto n = std::min(m.getShape()[0], m.getShape()[1]);
                Matrix<dType> r({n}, 0.,m.getData());
                for (msize_type i = 0; i < n; ++i)
                    r(i) = m(i, i);
                return r;
            }
        }

        template<Calculable dType>
        auto eye(const msize_type &n) -> Matrix<dType> {
            if (n <= 0)
                throw_value_error("eye构造矩阵维度长度至少大于0");
            Matrix<dType> r({n, n}, 0.);
            for (msize_type i = 0; i < n; ++i)
                r(i, i) = 1;
            return r;
        }

        template<Calculable dType>
        auto eye(const msize_type &m, const msize_type &n) -> Matrix<dType> {
            if (m < 0 || n < 0)
                throw_value_error("构造矩阵维度长度至少大于等于0");
            msize_type min_ = std::min(m, n);
            Matrix<dType> r({m, n}, 0.);
            for (msize_type i = 0; i < min_; ++i)
                r(i, i) = 1;
            return r;
        }

        auto ones() -> int_max {
            return 1;
        }

        auto ones(const int64 &n) -> Matrix<int_max> {
            return Matrix<int_max>({n, n}, 1);
        }

        auto ones(const initializer_list<msize_type> &shape) -> Matrix<int_max> {
            return Matrix<int_max>(mshape_type(shape), 1);
        }

        auto ones(const mshape_type &shape) -> Matrix<int_max> {
            return Matrix<int_max>(shape, 1);
        }

        template<Calculable TargetType>
        auto ones() -> TargetType {
            return 1.;
        }

        template<Calculable dType>
        auto ones(const msize_type &n) -> Matrix<dType> {
            return Matrix<dType>({n, n}, 1);
        }

        template<Calculable dType>
        auto ones(const mshape_type &shape) -> Matrix<dType> {
            return Matrix<dType>(shape, 1);
        }

        auto zeros() -> int_max {
            return 0;
        }

        auto zeros(const msize_type &n) -> Matrix<int_max> {
            return Matrix<int_max>({n, n}, 0.);
        }

        auto zeros(const mshape_type &shape) -> Matrix<int_max> {
            return Matrix<int_max>(shape, 0.);
        }

        template<Calculable TargetType>
        auto zeros() -> TargetType {
            return 0.;
        }

        template<Calculable dType>
        auto zeros(const msize_type &n) -> Matrix<dType> {
            return Matrix<dType>({n, n}, 0.);
        }

        template<Calculable dType>
        auto zeros(const mshape_type &shape) -> Matrix<dType> {
            return Matrix<dType>(shape, 0.);
        }
    }

    ///矩阵结构函数
    namespace Math {
        /**
         * 翻转元素顺序
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<Calculable dType>
        auto flip(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<dType> {
            msize_type dim_idx = dim - 1;
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0) {
                Matrix<dType> r(m.getShape(), 0.,m.getData());
                auto riter_m = m.getDataEnd() - 1;
                for (auto &i: r)
                    i = *riter_m--;
                return r;
            }
            Matrix<dType> r(m);
            msize_type step_size = Util::VectorUtil::mul(m.getShape(), 0, dim_idx);
            msize_type step_num = m.getShape()[dim_idx];
            msize_type unit_size = step_size * step_num;
            msize_type unit_num = m.getSize() / unit_size;
            unique_ptr<dType[]> tmp(new dType[step_num]);
            auto iter = r.getData();
            for (msize_type n = 0; n < unit_num; ++n) {
                for (msize_type i = 0; i < step_size; ++i) {
                    for (msize_type j = 0; j < step_num; ++j)
                        tmp[j] = *(iter + i + j * step_size);
                    for (msize_type j = 0; j < step_num; ++j)
                        *(iter + i + j * step_size) = tmp[step_num - 1 - j];
                }
                iter += unit_size;
            }
            return r;
        }

        /**
         * 左右指的是经典行列表达方法中的列向量互相顺序镜像
         * 即第二维中的一维向量倒置顺序
         */
        template<Calculable dType>
        auto fliplr(const Matrix<dType> &m) -> Matrix<dType> {
            if (m.getDims() < 2)
                return m;
            Matrix<dType> r(m.getShape(), 0.,m.getData());
            msize_type vec_size = m.getShape()[0];
            msize_type vec_num = m.getShape()[1];
            msize_type unit_size = vec_size * vec_num;
            msize_type unit_num = m.getSize() / unit_size;
            auto iter_r = r.getData();
            auto iter_m = m.getData();
            for (msize_type i = 0; i < unit_num; ++i) {
                for (msize_type j = 0; j < vec_num; ++j) {
                    for (msize_type k = 0; k < vec_size; ++k) {
                        *(iter_r++) = *(iter_m + vec_size * (vec_num - 1 - j) + k);
                    }
                }
                iter_m += unit_size;
            }
            return r;
        }

        /**
         * 上下指的是经典行列表达方法中的列向量内部左右镜像
         * 即第一维向量内部倒序\n
         * 等同于flip(Matrix,1)
         */
        template<Calculable dType>
        auto flipud(const Matrix<dType> &m) -> Matrix<dType> {
            return flip(m, 1);
        }


        /**
         * 逆时针旋转k*90度
         */
        template<Calculable dType>
        auto rot90(const Matrix<dType> &m, msize_type k = 1) -> Matrix<dType> {
            k = k % 4;
            k = k < 0 ? k + 4 : k;
            if (k == 0)
                return {m};
            else if (k == 1)
                return flip(m.T(), 1);
            else if (k == 2)
                return flip(flip(m).T()).T();
            else
                return flip(m).T();
        }

        /**
         * @param k 要包括的对角线，指定为标量。k = 0 是主对角线，k > 0 位于主对角线上方，而 k < 0 位于主对角线下方。
         */
        template<Calculable dType>
        auto tril(const Matrix<dType> &m, msize_type k = 0) -> Matrix<dType> {
            Matrix<dType> r(m);
            auto unit_shape = r.getUnitShape();
            for (msize_type n = 0; n < r.getUnitNum(); ++n) {
                for (msize_type i = 0; i < unit_shape[0]; ++i) {
                    for (msize_type j = 0; j < unit_shape[1]; ++j) {
                        if (j > i + k)
                            r(n, i, j) = 0;
                    }
                }
            }
            return r;
        }

        /**
         * @param k 要包括的对角线，指定为标量。k = 0 是主对角线，k > 0 位于主对角线上方，而 k < 0 位于主对角线下方。
         */
        template<Calculable dType>
        auto triu(const Matrix<dType> &m, msize_type k = 0) -> Matrix<dType> {
            Matrix<dType> r(m);
            auto unit_shape = r.getUnitShape();
            for (msize_type n = 0; n < r.getUnitNum(); ++n) {
                for (msize_type i = 0; i < unit_shape[0]; ++i) {
                    for (msize_type j = 0; j < unit_shape[1]; ++j) {
                        if (j < i + k)
                            r(n, i, j) = 0;
                    }
                }
            }
            return r;
        }

        /**
         * @param dim 沿着哪个维度进行排序 为负或0则全局排序，超过维度长度时范围返回矩阵本身
         * @param up 默认升序 false降序
         */
        template<Calculable dType>
        auto sort(const Matrix<dType> &m, msize_type dim = 1, bool up = true) -> Matrix<dType> {
            msize_type dim_idx = dim - 1;
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0)
                return m.sort(up);
            Matrix<dType> r(m);
            msize_type step_size = Util::VectorUtil::mul(m.getShape(), 0, dim_idx);
            msize_type step_num = m.getShape()[dim_idx];
            msize_type unit_size = step_size * step_num;
            msize_type unit_num = m.getSize() / unit_size;
            unique_ptr<dType[]> tmp(new dType[step_num]);
            auto iter = r.getData();
            for (msize_type n = 0; n < unit_num; ++n) {
                for (msize_type i = 0; i < step_size; ++i) {
                    for (msize_type j = 0; j < step_num; ++j)
                        tmp[j] = *(iter + i + j * step_size);
                    std::sort(tmp.get(), tmp.get() + step_num);
                    for (msize_type j = 0; j < step_num; ++j)
                        if (up)
                            *(iter + i + j * step_size) = tmp[j];
                        else
                            *(iter + i + j * step_size) = tmp[step_num - j - 1];
                }
                iter += unit_size;
            }
            return r;
        }

    }
    ///矩阵数学求值
    namespace Math {
        /**
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
         */
        template<Calculable dType>
        auto cumsum(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<dType> {
            msize_type dim_idx = dim - 1;
            if (m.isEmpty())
                return {};
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0) {
                Matrix<dType> r(m);
                dType sum = 0.;
                for (auto &i: r)
                    i = sum += i;
                return r;
            }
            Matrix<dType> r(m);
            msize_type step_size = Util::VectorUtil::mul(m.getShape(), 0, dim_idx);
            msize_type step_num = m.getShape()[dim_idx];
            msize_type unit_size = step_size * step_num;
            msize_type unit_num = m.getSize() / unit_size;
            auto iter = r.getData();
            for (msize_type n = 0; n < unit_num; ++n) {
                for (msize_type i = 0; i < step_size; ++i) {
                    dType sum = 0.;
                    for (msize_type j = 0; j < step_num; ++j) {
                        auto &item = *(iter + i + j * step_size);
                        item = sum += item;
                    }
                }
                iter += unit_size;
            }
            return r;
        }

        /**
         * geomean求n个数的乘积再开n次根
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
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
            msize_type step_size = MatSAR::Util::VectorUtil::mul(shape, 0, dim_idx);
            msize_type unit_size = step_size * shape[dim_idx];
            msize_type step_num = shape[dim_idx];
            msize_type unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<real_max> r(r_shape, 1,m.getData());//乘积需要初始化为1

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
         * 调和平均数
         * @param dim 为负或0时进行全局求值，在维度范围内返回沿着该维度的求值，超过维度长度时范围返回矩阵本身
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
            auto step_size = MatSAR::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = 1;
            Matrix<real_max> r(r_shape, 1,m.getData());

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
                throw_value_error("trace要求矩阵为方阵并且维度为2");
            dType r = 0;
            for (msize_type i = 0; i < m.getShape()[0]; ++i)
                r += m(i, i);
            return r;
        }
    }

    ///LU分解
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
                throw_value_error("svd需要二维矩阵或可扩展为二维矩阵");
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
            Matrix<int64> P({min_m_n, min_m_n}, 0.,M.getData());
            for (int i = 0; i < min_m_n; ++i)
                P(IDX[i], i) = 1;//P.T()!!!

//            cout << fm.toRowColString("fm") << endl;
//            cout << P.toString("p") << endl;
            if (m == n)
                return {P * (tril(fm, -1) + eye<TargetType>(m, n)), triu(fm)};
            else if (m > n) {
                auto U = (triu(fm))({Range(n), Range(n)});
                return {P * (tril(fm, -1) + eye<TargetType>(m, n)), U};
            } else {
                auto L = (P * (tril(fm, -1) + eye<TargetType>(m, n)))({Range(m), Range(m)});
                return {L, triu(fm)};
            }
        }

        template<RealType RT>
        auto lu(const Matrix<RT> &M) -> tuple<Matrix<float64>, Matrix<float64>> {
            return lu<float64>(M);
        }

        auto lu(const Matrix<complex64> &M) -> tuple<Matrix<complex64>, Matrix<complex64>> {
            return lu<complex64>(M);
        }

        auto lu(const Matrix<complex128> &M) -> tuple<Matrix<complex128>, Matrix<complex128>> {
            return lu<complex128>(M);
        }

        template<Calculable TargetType, Calculable dType>
        auto lup(const Matrix<dType> &M) -> tuple<Matrix<TargetType>, Matrix<TargetType>, Matrix<int64>> {
            Matrix<TargetType> fm(M);
            fm.expandShape(2);
            if (!fm.is2D())
                throw_value_error("svd需要二维矩阵或可扩展为二维矩阵");
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
            Matrix<int64> P({min_m_n, min_m_n}, 0.,M.getData());
            for (int i = 0; i < min_m_n; ++i)
                P(i, IDX[i]) = 1;//P!!!
//            cout << fm.toRowColString("fm") << endl;
//            return {(tril(fm, -1) + eye<TargetType>(m, n)), triu(fm), P};
            if (m == n)
                return {(tril(fm, -1) + eye<TargetType>(m, n)), triu(fm), P};
            else if (m > n) {
                auto U = (triu(fm))({Range(n), Range(n)});
                return {tril(fm, -1) + eye<TargetType>(m, n), U, P};
            } else {
                auto L = (tril(fm, -1) + eye<TargetType>(m, n))({Range(m), Range(m)});
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
        template<RealType RT>
        auto inv(const Matrix<RT> &m) -> Matrix<real_max> {
            if (!m.isSquare() || m.getDims() != 2)
                throw_value_error("矩阵求逆需要二维方阵");
            auto n = m.getShape()[0];
            Matrix<real_max> A(m);
            Matrix<real_max> X({n, n}, 0.,m.getData());

            lapack_int iter;

            unique_ptr<lapack_int[]> IPIV(new lapack_int[n]);

            auto info = LAPACKE_dsgesv(CblasColMajor, n, n, A.getData(), n,
                                       IPIV.get(), eye<real_max>(n).getData(), n, X.getData(), n, &iter);
            if (info < 0)
                throw_value_error("inv参数出错"+ to_string(info));
            if (info > 0){
                stringstream ss;
                ss<<"该矩阵秩为"<<rank(m)<<"形状为"<<n<<"无逆矩阵解";
                m.debug();
                throw_warning(ss.str());
            }
            return X;
        }

        template<ComplexType CT>
        auto inv(const Matrix<CT> &m) -> Matrix<complex_max> {
            if (!m.isSquare() || m.getDims() != 2)
                throw_value_error("矩阵求逆需要二维方阵");
            auto n = m.getShape()[0];
            Matrix<complex_max> A(m);
            Matrix<complex_max> X({n, n}, 0.,m.getData());

            lapack_int iter;

            unique_ptr<lapack_int[]> IPIV(new lapack_int[n]);

            auto info = LAPACKE_zcgesv(CblasColMajor, n, n, A.getData(), n,
                                       IPIV.get(), eye<complex_max>(n).getData(), n, X.getData(), n, &iter);
            if (info < 0)
                throw_value_error("inv参数出错"+ to_string(info));
            if (info > 0){
                stringstream ss;
                ss<<"该矩阵秩为"<<rank(m)<<"形状为"<<n<<"无逆矩阵解";
                m.debug();
                throw_warning(ss.str());
            }
            return X;
        }
    }
}

#endif //MATSAR_LINEARALGEBRA_HPP
