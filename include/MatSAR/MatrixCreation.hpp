//
// Created by JakoError on 24-7-31.
//

#ifndef MATSAR_MATRIXCREATION_HPP
#define MATSAR_MATRIXCREATION_HPP

#ifndef MATSAR_MATRIX_HPP

#include "./Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

#include "util/Compat.hpp"

namespace matsar {
    template<typename dType = float_max>
    auto zeros() -> dType {
        return 0;
    }

    template<typename dType = float_max>
    auto zeros(const mshape_type &shape) -> Matrix<dType> {
        return Matrix<dType>(shape, dType(0));
    }

    template<typename TargetType, typename dType, class = typename std::enable_if<!std::is_same<dType, TargetType>::value>::type>
    auto zeros_like(const Matrix<dType> &m,
                    MATSAR_UNUSED TargetType data_type = TargetType()) -> Matrix<TargetType> {
        return Matrix<TargetType>(m.getShape(), TargetType(0), m.getData());
    }

    template<typename dType>
    auto zeros_like(const Matrix<dType> &m) -> Matrix<dType> {
        return Matrix<dType>(m.getShape(), dType(0), m.getData());
    }

    template<typename dType = float_max>
    auto ones() -> dType {
        return 1;
    }

    template<typename dType = float_max>
    auto ones(const mshape_type &shape) -> Matrix<dType> {
        return Matrix<dType>(shape, dType(1));
    }

    template<typename TargetType, typename dType, class = typename std::enable_if<!std::is_same<dType, TargetType>::value>::type>
    auto ones_like(const Matrix<dType> &m,
                   MATSAR_UNUSED TargetType data_type = TargetType()) -> Matrix<TargetType> {
        return Matrix<TargetType>(m.getShape(), TargetType(1), m.getData());
    }

    template<typename dType>
    auto ones_like(const Matrix<dType> &m) -> Matrix<dType> {
        return Matrix<dType>(m.getShape(), dType(1), m.getData());
    }


    /**
     * If the matrix is 1-dimensional, it is expanded to 2-dimensional.
     * If the matrix is 2-dimensional, it is extracted diagonally to 1-dimensional.
     */
    template<typename dType>
    auto diag(const Matrix<dType> &m, const msize_type &k = 0) -> Matrix<dType> {
        if (m.isEmpty()) {
            return {};
        }
        if (m.getDims() > 2) {
            throw_value_error("diag: function requires the matrix to be 2D or 1D");
        }
        if (m.isVec()) {
            auto n = m.getShape()[0];
            auto rn = n + abs(k);

            auto r = zeros<dType>({rn, rn});
            for (msize_type i = 0; i < n; ++i) {
                if (k >= 0) {
                    r(i, i + k) = m(i);
                } else {
                    r(i - k, i) = m(i);
                }
            }
            return r;
        } else {
            auto n = std::min(m.getShape()[0], m.getShape()[1]);
            msize_type i_start = 0, i_end = n;
            if (k > 0) {
                if (i_end + k >= m.getShape()[1]) {
                    i_end = m.getShape()[1] - k;
                }
            } else if (k < 0) {
                i_start -= k;
            }

            if (i_start >= i_end) {
                return {};
            }

            auto r = zeros<dType>({i_end - i_start});
            for (msize_type i = i_start; i < i_end; ++i) {
                r(i) = m(i, i + k);
            }
            return r;
        }
    }

    template<typename dType = float_max>
    auto eye(const msize_type &n, Container<msize_type> m = none, const msize_type &k = 0) -> Matrix<dType> {
        if (m == none) {
            m = n;
        }

        auto r = zeros<dType>({n, m});

        msize_type i_start = 0, i_end = std::min(msize_type(n), msize_type(m));
        if (k > 0) {
            if (i_end + k >= m) {
                i_end = msize_type(m) - k;
            }
        } else if (k < 0) {
            i_start -= k;
        }

        for (msize_type i = i_start; i < i_end; ++i) {
            r(i, i + k) = dType(1);
        }
        return r;
    }
}

namespace matsar {
    template<typename T>
    auto arange_cal_size(const T &start, const T &stop, const T &step) -> msize_type = delete;

    template<>
    auto arange_cal_size<float64>(const float64 &start, const float64 &stop, const float64 &step) -> msize_type {
        return static_cast<msize_type>(std::ceil((stop - start) / step));
    }

    template<>
    auto arange_cal_size<int64>(const int64 &start, const int64 &stop, const int64 &step) -> msize_type {
        if ((stop - start) % step != 0) {
            return static_cast<msize_type>((stop - start) / step + 1);
        } else {
            return static_cast<msize_type>((stop - start) / step);
        }
    }

    template<RealType RT CheckReal(RT)>
    auto arange(const RT &start, const RT &stop, const RT &step) -> Matrix<RT> {
        if (start >= stop - step) {
            return {};
        }
        auto r_size = arange_cal_size(start, stop, step);
        auto r = zeros<RT>({r_size});
        RT value = start;
        auto iter_r = r.begin();
        for (decltype(r_size) i = 0; i < r_size; ++i) {
            *iter_r = value;
            value += step;
            iter_r++;
        }
        return r;
    }

    template<RealType RT CheckReal(RT)>
    auto arange(const RT &start, const RT &stop) -> Matrix<RT> {
        return arange<RT>(start, stop, RT(1));
    }

    template<RealType RT CheckReal(RT)>
    auto arange(const RT &stop) -> Matrix<RT> {
        return arange<RT>(RT(0), stop, RT(1));
    }

}

//Matrix structure function
namespace matsar {
    template<typename dType>
    auto flip(const Matrix<dType> &m, msize_type dim = 1) -> Matrix<dType> {
        msize_type dim_idx = dim - 1;
        if (dim_idx >= m.getDims())
            return {m};
        if (dim_idx < 0) {
            Matrix<dType> r(m.getShape(), 0., m.getData());
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
                for (msize_type j = 0; j < step_num / 2; ++j) {
                    std::swap(*(iter + i + j * step_size), *(iter + i + (step_num - j - 1) * step_size));
                }
            }
            iter += unit_size;
        }
        return r;
    }

    /**
     * 左右指的是经典行列表达方法中的列向量互相顺序镜像
     * 即第二维中的一维向量倒置顺序
     */
    template<typename dType>
    auto fliplr(const Matrix<dType> &m) -> Matrix<dType> {
        if (m.getDims() < 2)
            return m;
        Matrix<dType> r(m.getShape(), 0., m.getData());
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
    template<typename dType>
    auto flipud(const Matrix<dType> &m) -> Matrix<dType> {
        return flip(m, 1);
    }


    /**
     * 逆时针旋转k*90度
     */
    template<typename dType>
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
    template<typename dType>
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
    template<typename dType>
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
}

#endif //MATSAR_MATRIXCREATION_HPP
