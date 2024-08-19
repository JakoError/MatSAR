//
// Created by JakoError on 24-7-31.
//

#ifndef MATSAR_UNIVERSALFUNCTIONS_HPP
#define MATSAR_UNIVERSALFUNCTIONS_HPP

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

namespace MatSAR {
    template<typename T, typename TargetType=decltype(fun(std::declval<T>())), typename F>
    auto apply_function(const Matrix<T> &m, F fun) -> Matrix<TargetType> {
        Matrix<TargetType> r(m.getShape(), TargetType(0), m.getData());
        auto iter_m = m.begin();
        for (auto &i: r) {
            i = fun(*iter_m);
            iter_m++;
        }
        return r;
    }

    template<typename T1, typename T2, typename TargetType=decltype(fun(std::declval<T1>(),
                                                                        std::declval<T2>())), typename F>
    auto apply_function(const Matrix<T1> &m1, const Matrix<T2> &m2, F fun) -> Matrix<TargetType> {
        if (m1.getShape() != m2.getShape()) {
            throw_shape_exception(
                    "The two matrices have different shapes: m1 shape = " + m1.shapeToString() +
                    ", m2 shape = " + m2.shapeToString());
        }
        Matrix<TargetType> r(m1.getShape(), TargetType(0), m1.getData());
        auto iter_m1 = m1.begin();
        auto iter_m2 = m2.begin();
        for (auto &i: r) {
            i = fun(*iter_m1, *iter_m2);
            iter_m1++;
            iter_m2++;
        }
        return r;
    }

    template<typename T, typename TargetType, typename F>
    auto apply_function_global(const Matrix<T> &m, F fun, const TargetType &init_value = 0) -> Matrix<TargetType> {
        TargetType r = init_value;
        for (const auto &v: m) {
            r = fun(r, v);
        }
        return {r};
    }

    template<typename T, typename TargetType, typename F>
    auto apply_function_global_nn(const Matrix<T> &m, F fun, const TargetType &init_value = 0) -> Matrix<TargetType> {
        Matrix<TargetType> r(m.getShape(), init_value, m.getData());
        auto iter_r = r.begin();
        for (const auto &v: m) {
            *iter_r = fun(*iter_r, v);
            iter_r++;
        }
        return r;
    }

    template<typename T, typename TargetType, typename F>
    auto apply_function_global_nn_idx(const Matrix<T> &m, F fun_index,
                                      const TargetType &init_value = 0) -> Matrix<TargetType> {
        Matrix<TargetType> r(m.getShape(), init_value, m.getData());
        auto iter_m = m.begin();
        auto iter_r = r.begin();
        for (typename Matrix<T>::size_type i = 0; i < m.getSize(); ++i) {
            *iter_r = fun_index(*iter_r, *iter_m, i);
        }
        return r;
    }

    template<typename T, typename TargetType, typename F>
    auto apply_function_global_nk(const Matrix<T> &m, F fun_1d, const msize_type &k,
                                  const size_t &work_size = 0) -> Matrix<TargetType> {
        if (k <= 0) {
            return {};
        }
        Matrix<TargetType> r(m.getShape(), TargetType(0), m.getData());
        unique_ptr<T[]> src(new T[m.getSize()]);
        unique_ptr<TargetType[]> dst(new TargetType[k]);
        unique_ptr<T[]> work; // NULL
        if (work_size != 0) {
            work = unique_ptr<T[]>(new T[work_size]);
        }
        std::copy(m.begin(), m.end(), src.get());
        fun_1d(src.get(), dst.get(), k, work.get());
        std::copy_n(dst.get(), k, r.begin());
        return r;
    }

    template<typename T, typename TargetType, typename F>
    auto apply_function_n1(const Matrix<T> &m, F fun_1d, const msize_type &dim_idx,
                           const TargetType &init_value = 0) -> Matrix<TargetType> {
        auto shape = m.getShape();
        auto step_size = Util::VectorUtil::mul(shape, 0, dim_idx);
        auto step_num = shape[dim_idx];
        auto unit_size = step_size * step_num;
        auto unit_num = m.getSize() / unit_size;

        auto r_shape = shape;
        r_shape[dim_idx] = 1;
        Matrix<TargetType> r(r_shape, init_value, m.getData());

        auto iter_r = r.begin();
        auto iter_m = m.begin();

        for (msize_type i = 0; i < unit_num; ++i) {
            for (msize_type j = 0; j < step_num; ++j) {
                for (msize_type k = 0; k < step_size; ++k) {
                    *(iter_r + k) = fun_1d(*(iter_r + k), *(iter_m + k));
                }
                iter_m += step_size;
            }
            iter_r += step_size;
        }
        return r;
    }

    template<typename T, typename TargetType, typename F>
    auto apply_function_nn_idx(const Matrix<T> &m, F fun_index, const msize_type &dim_idx,
                               const TargetType &init_value = 0) -> Matrix<TargetType> {
        auto shape = m.getShape();
        auto step_size = Util::VectorUtil::mul(shape, 0, dim_idx);
        auto step_num = shape[dim_idx];
        auto unit_size = step_size * step_num;
        auto unit_num = m.getSize() / unit_size;

        auto r_shape = shape;
        Matrix<TargetType> r(r_shape, init_value, m.getData());

        auto iter_r = r.begin();
        auto iter_m = m.begin();

        for (msize_type n = 0; n < unit_num; ++n) {
            for (msize_type i = 0; i < step_size; ++i) {
                for (msize_type j = 0; j < step_num; ++j) {
                    auto &r_v = iter_r[i + j * step_size];
                    auto &m_v = iter_m[i + j * step_size];
                    r_v = fun_index(r_v, m_v, j);
                }
            }
            iter_m += unit_size;
            iter_r += unit_size;
        }
        return r;
    }

    template<typename T, typename TargetType, typename F>
    auto apply_function_nk(const Matrix<T> &m, F fun_1d, const msize_type &dim_idx, const msize_type &k,
                           const size_t &work_size = 0) -> Matrix<TargetType> {
        if (k <= 0) {
            return {};
        }
        auto shape = m.getShape();
        auto step_size = Util::VectorUtil::mul(shape, 0, dim_idx);
        auto step_num = shape[dim_idx];
        auto unit_size = step_size * step_num;
        auto unit_num = m.getSize() / unit_size;

        auto r_shape = shape;
        r_shape[dim_idx] = k;
        Matrix<TargetType> r(r_shape, TargetType(0), m.getData());
        auto r_unit_size = step_size * k;

        auto iter_r = r.begin();
        auto iter_m = m.begin();

        unique_ptr<T[]> src(new T[step_num]);
        unique_ptr<TargetType[]> dst(new TargetType[k]);
        unique_ptr<T[]> work;
        if (work_size != 0) {
            work = unique_ptr<T[]>(new T[work_size]);
        }
        for (msize_type n = 0; n < unit_num; ++n) {
            for (msize_type i = 0; i < step_size; ++i) {
                for (msize_type j = 0; j < step_num; ++j) {
                    src[j] = iter_m[i + j * step_size];
                }
                fun_1d(src.get(), dst.get(), k, work.get());
                for (msize_type j = 0; j < k; ++j) {
                    iter_r[i + j * step_size] = dst[j];
                }
            }
            iter_m += unit_size;
            iter_r += r_unit_size;
        }
        return r;
    }

//    template<typename T, typename TargetType, typename F, typename ...Args>
//    auto apply_function_2d(const Matrix<T> &m, F fun_2d, const mshape_type &shape, Args... args) -> Matrix<TargetType> {
//        Matrix<TargetType> r(shape, TargetType(0), m.getData());
//        auto m_unit_size = m.getUnitSize();
//        auto r_unit_size = r.getUnitSize();
//        auto m_unit_num = m.getUnitNum();
//        assert(m_unit_num == r.getUnitNum());
//        auto m_iter = m.begin();
//        auto r_iter = r.begin();
//
//        auto n = m_unit_num;
//        while (n-- > 0) {
//            fun_2d(m_iter, r_iter, &args...);
//            m_iter += m_unit_size;
//            r_iter += r_unit_size;
//        }
//        return r;
//    }
//
//    template<typename T, typename TargetType, typename F, typename ...Args>
//    auto apply_function_2d(const Matrix<T> &m, F fun_2d, const vector<mshape_type> &shapes,
//                           Args... args) -> vector<Matrix<TargetType>> {
//        auto m_unit_size = m.getUnitSize();
//        auto m_unit_num = m.getUnitNum();
//        auto r_len = shapes.size();
//        vector<Matrix<TargetType>> rs(r_len);
//        for (typeof(r_len) i = 0; i < r_len; ++i) {
//            rs[i] = Matrix<TargetType>(shapes[i], TargetType(0), m.getData());
//            assert(m_unit_num == rs[i].getUnitNum());
//        }
//        vector<typeof(Matrix<TargetType>::begin())> r_iters(r_len);
//        vector<typeof(Matrix<TargetType>::getUnitSize())> r_unit_sizes(r_len);
//        for (typeof(r_len) i = 0; i < r_len; ++i) {
//            r_iters[i] = rs[i].begin();
//            r_unit_sizes[i] = rs[i].getUnitSize();
//        }
//
//        auto m_iter = m.begin();
//
//        auto n = m_unit_num;
//        while (n-- > 0) {
//            fun_2d(m_iter, r_iters, &args...);
//            m_iter += m_unit_size;
//            for (typeof(r_len) i = 0; i < r_len; ++i) {
//                r_iters[i] += r_unit_sizes[i];
//            }
//        }
//        return rs;
//    }

//    template<typename T, typename F, typename ...Ts, typename F_HANDLE, typename ...Args>
//    auto apply_function_2d_p(const Matrix<T> &m, F fun_2d,
//                             const tuple<Ts...> &r_iters, const vector<msize_type> &r_unit_sizes,
//                             F_HANDLE f_handle,
//                             Args... args) -> void {
//        auto m_unit_size = m.getUnitSize();
//        auto m_unit_num = m.getUnitNum();
//        auto m_iter = m.begin();
//        auto r_len = r_unit_sizes.size();
//
//        auto n = m_unit_num;
//        while (n-- > 0) {
//            auto info = fun_2d(m_iter, r_iters, &args...);
//            f_handle(info);
//            m_iter += m_unit_size;
//            for (typeof(r_len) i = 0; i < r_len; ++i) {
//                auto &iter = get<i>(r_iters);
//                if (iter != nullptr) {
//                    iter += r_unit_sizes[i];
//                }
//            }
//        }
//    }
}

#endif //MATSAR_UNIVERSALFUNCTIONS_HPP
