//
// Created by JakoError on 24-7-31.
//

#ifndef MATSAR_UNIVERSALFUNCTIONS_HPP
#define MATSAR_UNIVERSALFUNCTIONS_HPP

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

namespace MatSAR {
    namespace Internal {
        template<typename T>
        class DimIter : public std::iterator<std::bidirectional_iterator_tag, T> {
        public:
            T *iter = nullptr;
            msize_type step_size = 1;

            DimIter(T *iter,
                    const msize_type &stepSize) : iter(iter), step_size(stepSize) {};

            auto inline operator[](const msize_type &i) -> T & {
                return iter[i * step_size];
            }

            auto inline operator*() -> T & {
                return *iter;
            }

            auto inline operator=(const DimIter<T> &another) -> DimIter<T> & = default;

            auto inline operator++() -> DimIter<T> & {
                this->iter += step_size;
                return *this;
            }

            auto inline operator--() -> DimIter<T> & {
                this->iter -= step_size;
                return *this;
            }

            auto inline operator+=(size_t i) -> DimIter<T> & {
                this->iter += i * step_size;
                return *this;
            }

            auto inline operator-=(size_t i) -> DimIter<T> & {
                this->iter -= i * step_size;
                return *this;
            }

            auto inline operator+(size_t i) -> DimIter<T> {
                return {iter + i * step_size, step_size};
            }

            auto inline operator-(size_t i) -> DimIter<T> {
                return {iter - i * step_size, step_size};
            }

            auto inline operator-(const DimIter &rhs) -> typeof((this->iter - rhs.iter) / step_size) {
                return (this->iter - rhs.iter) / step_size;
            }

            bool operator==(const DimIter &rhs) const {
                return this->iter == rhs.iter;
            }

            bool operator!=(const DimIter &rhs) const {
                return this->iter != rhs.iter;
            }

            bool operator<(const DimIter &rhs) const {
                return this->iter < rhs.iter;
            }

            bool operator>(const DimIter &rhs) const {
                return this->iter > rhs.iter;
            }

            bool operator<=(const DimIter &rhs) const {
                return this->iter < rhs.iter;
            }

            bool operator>=(const DimIter &rhs) const {
                return this->iter > rhs.iter;
            }
        };

        template<typename T>
        auto makeDimIter(T *iter,
                         const msize_type &stepSize) -> DimIter<T> {
            return {iter, stepSize};
        }
    }

    /**
     * apply function is for single-wise function implementation,\n
     * just apply this function through all single elements in matrix.
     * @tparam T auto
     * @tparam TargetType auto
     * @tparam F auto
     * @param m matrix
     * @param fun lambda or functions like fun(T)
     * @return result matrix
     */
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

    /**
     * apply function is for single-wise function implementation,\n
     * just apply this function through all single-pair elements in 2 matrices.
     * @tparam T1 auto
     * @tparam T2 auto
     * @tparam TargetType auto
     * @tparam F auto
     * @param m1 matrix-1
     * @param m2 matrix-2
     * @param fun lambda or function like fun(T1, T2)
     * @return result matrix
     */
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

    /**
     * apply function global is for full size to full size function implementation,\n
     * just apply this function through all elements (independent) in matrix.
     * @tparam T auto
     * @tparam TargetType specify
     * @tparam F auto
     * @param m matrix
     * @param fun lambda or function like fun(TargetType, T)
     * @param init_value initial value for result matrix
     * @return result matrix
     */
    template<typename T, typename TargetType, typename F>
    auto
    apply_function_global_all_all(const Matrix<T> &m, F fun, const TargetType &init_value = 0) -> Matrix<TargetType> {
        Matrix<TargetType> r(m.getShape(), init_value, m.getData());
        auto iter_r = r.begin();
        for (const auto &v: m) {
            *iter_r = fun(*iter_r, v);
            iter_r++;
        }
        return r;
    }

    /**
     * apply function global idx is for full size to full size function implementation with index passing,\n
     * just apply this function through all elements (independent) in matrix.
     * @tparam T auto
     * @tparam TargetType specify
     * @tparam F auto
     * @param m matrix
     * @param fun lambda or function like fun(TargetType, T, (matrix size type)index)
     * @param init_value initial value for result matrix
     * @return result matrix
     */
    template<typename T, typename TargetType, typename F>
    auto apply_function_global_all_all_idx(const Matrix<T> &m, F fun_index,
                                           const TargetType &init_value = 0) -> Matrix<TargetType> {
        Matrix<TargetType> r(m.getShape(), init_value, m.getData());
        auto iter_m = m.begin();
        auto iter_r = r.begin();
        for (typename Matrix<T>::size_type i = 0; i < m.getSize(); ++i) {
            *iter_r = fun_index(*iter_r, *iter_m, i);
        }
        return r;
    }

    /**
     * apply function global n1 is for n to 1 function implementation,\n
     * just apply this function through all single elements in matrix and get a single result.
     * @tparam T auto
     * @tparam TargetType auto
     * @tparam F auto
     * @param m matrix
     * @param fun lambda or function like fun(TargetType, T)
     * @param init_value initial value for result matrix
     * @return result matrix
     */
    template<typename T, typename TargetType, typename F>
    auto
    apply_function_global_all_1(const Matrix<T> &m, F fun, const TargetType &init_value = 0) -> Matrix<TargetType> {
        TargetType r = init_value;
        for (const auto &v: m) {
            r = fun(r, v);
        }
        return {r};
    }

    template<typename T, typename TargetType, typename WorkType=T, typename F>
    auto apply_function_global_all_k(const Matrix<T> &m, F fun_1d, const msize_type &k = 0,
                                     const size_t &work_size = 0, const bool &keep_shape = true) -> Matrix<TargetType> {
        auto r_shape = m.getShape();
        if (!keep_shape) {
            assert(k > 0);
            r_shape = {k};
        }
        Matrix<TargetType> r(r_shape, TargetType(0), m.getData());
        unique_ptr<WorkType[]> work; // default NULL
        if (work_size > 0) {
            work = unique_ptr<WorkType[]>(new WorkType[work_size]);
        }
        fun_1d(m.getData(), r.getData(), m.getSize(), k, work.get());
        return r;
    }

    template<typename T, typename TargetType, typename WorkType=T, typename F>
    auto apply_function_global_n_k(const Matrix<T> &m, F fun_1d, const msize_type &n = 0, const msize_type &k = 0,
                                   const size_t &work_size = 0, const bool &keep_shape = true) -> Matrix<TargetType> {
        assert(n <= m.getSize());
        auto r_shape = m.getShape();
        if (!keep_shape) {
            assert(k > 0);
            r_shape = {k};
        }
        Matrix<TargetType> r(r_shape, TargetType(0), m.getData());
        unique_ptr<WorkType[]> work; // default NULL
        if (work_size > 0) {
            work = unique_ptr<WorkType[]>(new WorkType[work_size]);
        }
        fun_1d(m.getData(), r.getData(), n, k, work.get());
        return r;
    }

    template<typename T, typename TargetType, typename WorkType=T, template<typename, typename, typename> class F>
    auto apply_tfunction_global_all_k(const Matrix<T> &m, const msize_type &k = 0,
                                      const size_t &work_size = 0,
                                      const bool &keep_shape = true) -> Matrix<TargetType> {
        // for only k specified the m rely on k to iter
        assert(k <= m.getSize());
        auto r_shape = m.getShape();
        if (!keep_shape) {
            assert(k > 0);
            r_shape = {k};
        }
        Matrix<TargetType> r(r_shape, TargetType(0), m.getData());
        unique_ptr<WorkType[]> work; // NULL
        if (work_size > 0) {
            work = unique_ptr<WorkType[]>(new WorkType[work_size]);
        }
        F<typeof(m.getData()), typeof(r.getData()), typeof(work.get())>()(m.getData(), r.getData(), k, work.get());
        return r;
    }

    template<typename T, typename TargetType, typename WorkType=T, template<typename, typename, typename> class F>
    auto apply_tfunction_global_n_k(const Matrix<T> &m, const msize_type &n = 0, const msize_type &k = 0,
                                    const size_t &work_size = 0, const bool &keep_shape = true) -> Matrix<TargetType> {
        auto r_shape = m.getShape();
        if (!keep_shape) {
            assert(k > 0);
            r_shape = {k};
        }
        Matrix<TargetType> r(r_shape, TargetType(0), m.getData());
        unique_ptr<WorkType[]> work; // NULL
        if (work_size > 0) {
            work = unique_ptr<T[]>(new WorkType[work_size]);
        }
        F<typeof(m.getData()), typeof(r.getData()), typeof(work.get())>()(m.getData(), r.getData(), n, k, work.get());
        return r;
    }

    template<typename T, typename TargetType, typename F>
    auto apply_function_all_1(const Matrix<T> &m, F fun_1d, const msize_type &dim_idx,
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
    auto apply_function_all_all(const Matrix<T> &m, F fun_index, const msize_type &dim_idx,
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

        for (msize_type u = 0; u < unit_num; ++u) {
            for (msize_type i = 0; i < step_size; ++i) {
                auto inner_iter_r = Internal::makeDimIter(iter_r + i, step_size);
                auto inner_iter_m = Internal::makeDimIter(iter_m + i, step_size);
                for (msize_type j = 0; j < step_num; ++j) {
                    auto &r_v = inner_iter_r[j];
                    auto &m_v = inner_iter_m[j];
                    r_v = fun_index(r_v, m_v);
                }
            }
            iter_m += unit_size;
            iter_r += unit_size;
        }
        return r;
    }

    template<typename T, typename TargetType, typename F>
    auto apply_function_all_all_idx(const Matrix<T> &m, F fun_index, const msize_type &dim_idx,
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

        for (msize_type u = 0; u < unit_num; ++u) {
            for (msize_type i = 0; i < step_size; ++i) {
                auto inner_iter_r = Internal::makeDimIter(iter_r + i, step_size);
                auto inner_iter_m = Internal::makeDimIter(iter_m + i, step_size);
                for (msize_type j = 0; j < step_num; ++j) {
                    auto &r_v = inner_iter_r[j];
                    auto &m_v = inner_iter_m[j];
                    r_v = fun_index(r_v, m_v, j);
                }
            }
            iter_m += unit_size;
            iter_r += unit_size;
        }
        return r;
    }

    template<typename T, typename TargetType, typename WorkType=T, typename F>
    auto apply_function_all_k(const Matrix<T> &m, F fun_1d, const msize_type &dim_idx, const msize_type &k,
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
        unique_ptr<WorkType[]> work;
        if (work_size > 0) {
            work = unique_ptr<WorkType[]>(new WorkType[work_size]);
        }
        for (msize_type u = 0; u < unit_num; ++u) {
            for (msize_type i = 0; i < step_size; ++i) {
                auto inner_iter_m = Internal::makeDimIter(iter_m + i, step_size);
                auto inner_iter_r = Internal::makeDimIter(iter_r + i, step_size);
                fun_1d(inner_iter_r, inner_iter_m, k, work.get());
            }
            iter_m += unit_size;
            iter_r += r_unit_size;
        }
        return r;
    }

    template<typename T, typename TargetType, typename WorkType=T, template<typename, typename, typename> class F>
    auto apply_tfunction_all_k(const Matrix<T> &m, const msize_type &dim_idx, const msize_type &k,
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
        auto r_unit_size = step_size * r.getShape()[dim_idx];

        auto iter_r = r.begin();
        auto iter_m = m.begin();

        unique_ptr<WorkType[]> work;
        if (work_size > 0) {
            work = unique_ptr<WorkType[]>(new WorkType[work_size]);
        }
        for (msize_type u = 0; u < unit_num; ++u) {
            for (msize_type i = 0; i < step_size; ++i) {
                auto inner_iter_m = Internal::makeDimIter(iter_m + i, step_size);
                auto inner_iter_r = Internal::makeDimIter(iter_r + i, step_size);
                F<typeof(inner_iter_m), typeof(inner_iter_r), typeof(work.get())>()(inner_iter_m, inner_iter_r, k,
                                                                                    work.get());
            }
            iter_m += unit_size;
            iter_r += r_unit_size;
        }
        return r;
    }

    template<typename T, typename TargetType, typename WorkType=T, template<typename, typename, typename> class F>
    auto apply_tfunction_n_k(const Matrix<T> &m, const msize_type &dim_idx, const msize_type &n, const msize_type &k,
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
        auto r_unit_size = step_size * r.getShape()[dim_idx];

        auto iter_r = r.begin();
        auto iter_m = m.begin();

        unique_ptr<WorkType[]> work;
        if (work_size > 0) {
            work = unique_ptr<WorkType[]>(new WorkType[work_size]);
        }
        for (msize_type u = 0; u < unit_num; ++u) {
            for (msize_type i = 0; i < step_size; ++i) {
                auto inner_iter_m = Internal::makeDimIter(iter_m + i, step_size);
                auto inner_iter_r = Internal::makeDimIter(iter_r + i, step_size);
                F<typeof(inner_iter_m), typeof(inner_iter_r), typeof(work.get())>()(inner_iter_m, inner_iter_r, n, k,
                                                                                    work.get());
            }
            iter_m += unit_size;
            iter_r += r_unit_size;
        }
        return r;
    }
//    template<typename T, typename TargetType, template<typename, typename, typename> class F>
//    auto apply_tfunction_nk(const Matrix<T> &m, const msize_type &dim_idx, const msize_type &n, const msize_type &k,
//                            const size_t &work_size = 0) -> Matrix<TargetType> {
//        if (k <= 0) {
//            return {};
//        }
//        auto shape = m.getShape();
//        auto step_size = Util::VectorUtil::mul(shape, 0, dim_idx);
//        auto step_num = shape[dim_idx];
//        auto unit_size = step_size * step_num;
//        auto unit_num = m.getSize() / unit_size;
//
//        auto r_shape = shape;
//        r_shape[dim_idx] = k;
//        Matrix<TargetType> r(r_shape, TargetType(0), m.getData());
//        auto r_unit_size = step_size * r.getShape()[dim_idx];
//
//        auto iter_r = r.begin();
//        auto iter_m = m.begin();
//
//        unique_ptr<T[]> work;
//        if (work_size > 0) {
//            work = unique_ptr<T[]>(new T[work_size]);
//        }
//        for (msize_type u = 0; u < unit_num; ++u) {
//            for (msize_type i = 0; i < step_size; ++i) {
//                auto inner_iter_m = Internal::makeDimIter(iter_m + i, step_size);
//                auto inner_iter_r = Internal::makeDimIter(iter_r + i, step_size);
//                F<typeof(inner_iter_r), typeof(inner_iter_m), typeof(work.get())>()(inner_iter_m, inner_iter_r, n, k,
//                                                                                    work.get());
//            }
//            iter_m += unit_size;
//            iter_r += r_unit_size;
//        }
//        return r;
//    }

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
