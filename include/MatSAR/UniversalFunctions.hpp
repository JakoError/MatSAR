// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 24-7-31.
//

#ifndef MATSAR_UNIVERSALFUNCTIONS_HPP
#define MATSAR_UNIVERSALFUNCTIONS_HPP

#ifndef MATSAR_MATRIX_HPP

#include "Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

// std::remove_cvref_t is C++20; reuse it there, backport it for C++17.
#if defined(__cpp_lib_remove_cvref)
using std::remove_cvref_t;
#else
template<class X>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<X>>;
#endif

namespace matsar {
    namespace Internal {
        template<typename T>
        class DimIter : public std::iterator<std::bidirectional_iterator_tag, T> {
        protected:
            T hidden_zero = 0;

        public:
            virtual ~DimIter() = default;

            T *iter = nullptr;
            msize_type step_size = 1;
            msize_type length = 0;
            msize_type position = 0;

//            std::unique_ptr<T[]> extended_space = nullptr;
//            msize_type extended_length;

//            DimIter(T *iter, const msize_type &stepSize, const msize_type &length,
//                    const msize_type &position = 0) : iter(iter), step_size(stepSize),
//                                                      position(position), length(length), extended_length(length) {};
//
//            DimIter(T *iter, const msize_type &stepSize, const msize_type &length,
//                    const msize_type &position, const msize_type &extendedLength) : iter(iter), step_size(stepSize),
//                                                      position(position), length(length), extended_length(length) {};

            DimIter(T *iter, const msize_type &stepSize, const msize_type &length,
                    const msize_type &position = 0) : iter(iter), step_size(stepSize),
                                                      length(length), position(position) {}

            virtual auto operator[](const msize_type &i) -> T & {
                auto control_i = position + i;
                if (control_i >= length || control_i < 0) {
                    if (hidden_zero != T(0)) {
                        throw_warning("For DimIter, the access of memory is protected even index is out of range. "
                                      "The protected zero is changed. Check the algorithm for more information.");
                        hidden_zero = T(0);
                    }
                    return hidden_zero;
                }
                return iter[i * step_size];
            }

            auto operator*() -> T & {
                return this->operator[](0);
            }

            auto operator=(const DimIter<T> &another) -> DimIter<T> & = default;

            auto operator++() -> DimIter<T> & {
                this->iter += step_size;
                this->position++;
                return *this;
            }

            auto operator--() -> DimIter<T> & {
                this->iter -= step_size;
                --this->position;
                return *this;
            }

            auto operator+=(size_t i) -> DimIter<T> & {
                this->iter += i * step_size;
                this->position += i;
                return *this;
            }

            auto inline operator-=(size_t i) -> DimIter<T> & {
                this->iter -= i * step_size;
                this->position -= i;
                return *this;
            }

            auto inline operator+(size_t i) -> DimIter<T> {
                DimIter<T> result = *this;
                result += i;
                return result;
            }

            auto inline operator-(size_t i) -> DimIter<T> {
                DimIter<T> result = *this;
                result -= i;
                return result;
            }

            auto inline operator-(const DimIter &rhs) -> decltype((this->iter - rhs.iter) / step_size) {
                assert(this->step_size == rhs.step_size);
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
                return this->iter <= rhs.iter;
            }

            bool operator>=(const DimIter &rhs) const {
                return this->iter >= rhs.iter;
            }
        };

//        template<typename T>
//        class DimIterExtend : DimIter<T> {
//        public:
//
//
//            auto inline operator[](const msize_type &i) -> T & {
//                auto control_i = this->position + i;
//                if (control_i >= extended_length || control_i < 0) {
//                    if (this->hidden_zero != T(0)) {
//                        throw_warning("For DimIter, the access of memory is protected even index is out of range. "
//                                      "The protected zero is changed. Check the algorithm for more information.");
//                        this->hidden_zero = T(0);
//                    }
//                    return this->hidden_zero;
//                }
//                if (control_i >= this->length) {
//                    auto extended_i = i - this->length;
//                    return extended_space[extended_i];
//                }
//                return this->iter[i * this->step_size];
//            }
//
//            DimIterExtend(T *iter, const msize_type &stepSize, const msize_type &length, const msize_type &position,
//                          const msize_type &extendedLength) : DimIter<T>(iter, stepSize, length, position),
//                                                              extended_length(extendedLength) {
//                assert(extendedLength >= length);
//                extended_space = std::unique_ptr<T[]>(new T[extendedLength - length]);
//            }
//        };

        template<typename T>
        auto makeDimIter(T *iter,
                         const msize_type &stepSize,
                         const msize_type &length,
                         const msize_type &position = 0) -> DimIter<T> {
            return {iter, stepSize, length, position};
        }

//        template<typename T>
//        auto makeDimIterExtend(T *iter,
//                               const msize_type &stepSize,
//                               const msize_type &length,
//                               const msize_type &position,
//                               const msize_type &extendedLength
//        ) -> DimIterExtend<T> {
//            return {iter, stepSize, length, position, extendedLength};
//        }
    }

    // ===== apply_* family =======================================================
    // Internal building blocks behind the public ops (sum, prod, all, any, sort,
    // fft, ...). Two kinds, by how the callback is supplied:
    //   - element appliers: a lambda on elements -- fun(x) or fun(acc, x).
    //         _map = N->N size-preserving,  _reduce = N->1 fold,  *_idx also gets index.
    //   - kernel appliers : a 1-D routine over raw data/iterators --
    //         fun_1d(src, dst, [n,] k, work) writing k outputs; the apply_tfunction_*
    //         variants take the kernel as a functor TYPE F<...> instead of a value.
    // Scope token: 'global' = whole flattened matrix; 'dim' = along dim_idx.
    // ===========================================================================

    /**
     * Element-wise map over one matrix: result(i) = fun(m(i)); shape preserved.
     * @param m input matrix
     * @param fun element map, fun(const T& x) -> result element
     * @return result matrix (same shape as m)
     */
    template<typename T, typename TargetType = void, typename F>
    auto apply_function(const Matrix<T> &m, F fun)
            -> Matrix<
                    std::conditional_t<
                            std::is_void_v<TargetType>,
                            remove_cvref_t<std::invoke_result_t<F &, const T &>>,
                            TargetType>> {
        using Out = std::conditional_t<
                std::is_void_v<TargetType>,
                remove_cvref_t<std::invoke_result_t<F &, const T &>>,
                TargetType>;
        static_assert(!std::is_void_v<Out>,
                      "apply_function cannot produce Matrix<void>");
        Matrix<Out> r(m.getShape(), Out(0), m.getData());
        auto iter_m = m.begin();
        for (auto &i: r) {
            i = fun(*iter_m);
            iter_m++;
        }
        return r;
    }

    /**
     * Element-wise map over two equally-shaped matrices: result(i) = fun(m1(i), m2(i)).
     * Throws if the shapes differ.
     * @param m1 first input matrix
     * @param m2 second input matrix
     * @param fun element map, fun(const T1& a, const T2& b) -> result element
     * @return result matrix (same shape as the inputs)
     */
    template<typename T1, typename T2, typename TargetType = void, typename F>
    auto apply_function(const Matrix<T1> &m1, const Matrix<T2> &m2, F fun)
            -> Matrix<
                    std::conditional_t<
                            std::is_void_v<TargetType>,
                            remove_cvref_t<std::invoke_result_t<F &, const T1 &, const T2 &>>,
                            TargetType>> {
        if (m1.getShape() != m2.getShape()) {
            throw_shape_exception(
                    "The two matrices have different shapes: m1 shape = " + m1.shapeToString() +
                    ", m2 shape = " + m2.shapeToString());
        }
        using Out = std::conditional_t<
                std::is_void_v<TargetType>,
                remove_cvref_t<std::invoke_result_t<F &, const T1 &, const T2 &>>,
                TargetType>;
        static_assert(!std::is_void_v<Out>,
                      "apply_function cannot produce Matrix<void>");
        Matrix<Out> r(m1.getShape(), Out(0), m1.getData());
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
     * Size-preserving map over the whole matrix: result(i) = fun(result(i), m(i)), with
     * each result element seeded to init_value. Elements are processed independently.
     * @param m input matrix
     * @param fun binary map, fun(const TargetType& acc, const T& x) -> result element
     * @param init_value initial value of each result element
     * @return result matrix (same shape as m)
     */
    template<typename T, typename TargetType = void, typename F, typename Init = int>
    auto apply_function_global_map(const Matrix<T> &m, F fun, const Init &init_value = 0)
            -> Matrix<std::conditional_t<
                    std::is_void_v<TargetType>,
                    remove_cvref_t<std::invoke_result_t<F &, const TargetType &, const T &>>,
                    TargetType>> {
        using Out = std::conditional_t<
                std::is_void_v<TargetType>,
                remove_cvref_t<std::invoke_result_t<F &, const TargetType &, const T &>>,
                TargetType>;
        static_assert(!std::is_void_v<Out>,
                      "apply_function_global_map cannot produce Matrix<void>");
        Matrix<Out> r(m.getShape(), Out(init_value), m.getData());
        auto iter_r = r.begin();
        for (const auto &v: m) {
            *iter_r = fun(*iter_r, v);
            iter_r++;
        }
        return r;
    }

    /**
     * Size-preserving map over the whole matrix with the flat index: result(i) =
     * fun_index(result(i), m(i), i), each result element seeded to init_value.
     * @param m input matrix
     * @param fun_index binary map with index, fun(const TargetType& acc, const T& x, size_type i)
     * @param init_value initial value of each result element
     * @return result matrix (same shape as m)
     */
    template<typename T, typename TargetType = void, typename F, typename Init = int>
    auto apply_function_global_map_idx(const Matrix<T> &m, F fun_index, const Init &init_value = 0)
            -> Matrix<std::conditional_t<
                    std::is_void_v<TargetType>,
                    remove_cvref_t<std::invoke_result_t<F &, const TargetType &, const T &, const typename Matrix<T>::size_type &>>,
                    TargetType>> {
        using Out = std::conditional_t<
                std::is_void_v<TargetType>,
                remove_cvref_t<std::invoke_result_t<F &, const TargetType &, const T &, const typename Matrix<T>::size_type &>>,
                TargetType>;
        static_assert(!std::is_void_v<Out>,
                      "apply_function_global_map_idx cannot produce Matrix<void>");
        Matrix<Out> r(m.getShape(), Out(init_value), m.getData());
        auto iter_m = m.begin();
        auto iter_r = r.begin();
        for (typename Matrix<T>::size_type i = 0; i < m.getSize(); ++i) {
            *iter_r = fun_index(*iter_r, *iter_m, i);
        }
        return r;
    }

    /**
     * Fold the whole matrix to a single value: r = fun(r, m(i)) over all elements,
     * starting from init_value (e.g. global sum or product).
     * @param m input matrix
     * @param fun binary fold, fun(const TargetType& acc, const T& x) -> new accumulator
     * @param init_value initial accumulator value
     * @return result matrix holding the single value (shape {1})
     */
    template<typename T, typename TargetType = void, typename F, typename Init>
    auto apply_function_global_reduce(const Matrix<T> &m, F fun, const Init &init_value = 0)
            -> Matrix<std::conditional_t<
                    std::is_void_v<TargetType>,
                    remove_cvref_t<std::invoke_result_t<F &, const TargetType &, const T &>>,
                    TargetType>> {
        using Out = std::conditional_t<
                std::is_void_v<TargetType>,
                remove_cvref_t<std::invoke_result_t<F &, const TargetType &, const T &>>,
                TargetType>;
        static_assert(!std::is_void_v<Out>,
                      "apply_function_global_reduce cannot produce Matrix<void>");
        Out r = init_value;
        for (const auto &v: m) {
            r = fun(r, v);
        }
        return {r};
    }

    /**
     * Apply a raw 1-D kernel over the whole (flattened) matrix, called once as
     * fun_1d(src, dst, n = m.getSize(), k, work): reads all source elements, writes k.
     * @tparam WorkType element type of the optional scratch buffer (default T)
     * @param m input matrix
     * @param fun_1d kernel, void(const T* src, TargetType* dst, msize_type n, msize_type k, WorkType* work)
     * @param k number of outputs the kernel writes
     * @param work_size if > 0, a WorkType[work_size] scratch buffer is passed as work
     * @param keep_shape true: keep m's shape; false: reshape result to {k}
     * @return result matrix
     */
    template<typename T, typename TargetType, typename WorkType = T, typename F>
    auto apply_function_global_k(const Matrix<T> &m, F fun_1d, const msize_type &k = 0,
                                 const size_t &work_size = 0, const bool &keep_shape = true)
            -> Matrix<TargetType> {
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

    /**
     * Like apply_function_global_k, but the kernel reads exactly n source elements
     * (n <= m.getSize()) instead of all: fun_1d(src, dst, n, k, work).
     * @tparam WorkType element type of the optional scratch buffer (default T)
     * @param m input matrix
     * @param fun_1d kernel, void(const T* src, TargetType* dst, msize_type n, msize_type k, WorkType* work)
     * @param n number of source elements the kernel consumes
     * @param k number of outputs the kernel writes
     * @param work_size if > 0, a WorkType[work_size] scratch buffer is passed as work
     * @param keep_shape true: keep m's shape; false: reshape result to {k}
     * @return result matrix
     */
    template<typename T, typename TargetType, typename WorkType = T, typename F>
    auto apply_function_global_n_k(const Matrix<T> &m, F fun_1d, const msize_type &n = 0, const msize_type &k = 0,
                                   const size_t &work_size = 0, const bool &keep_shape = true)
            -> Matrix<TargetType> {
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

    /**
     * Functor-type variant of apply_function_global_k: the kernel is a functor template F
     * (e.g. the optimized SORT/FFT/reduction structs), invoked over the whole matrix as
     * F<SrcPtr, DstPtr, WorkPtr>()(src, dst, k, work).
     * @tparam WorkType element type of the optional scratch buffer (default T)
     * @tparam F kernel functor template, called as F()(src, dst, k, work)
     * @param m input matrix
     * @param k number of outputs the kernel writes
     * @param work_size if > 0, a WorkType[work_size] scratch buffer is passed as work
     * @param keep_shape true: keep m's shape; false: reshape result to {k}
     * @return result matrix
     */
    template<typename T, typename TargetType, typename WorkType = T, template<typename, typename, typename> class F>
    auto apply_tfunction_global_k(const Matrix<T> &m, const msize_type &k = 0,
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
        F<decltype(m.getData()), decltype(r.getData()), decltype(work.get())>()(m.getData(), r.getData(), k, work.get());
        return r;
    }

    /**
     * Functor-type variant of apply_function_global_n_k: F()(src, dst, n, k, work) over the
     * whole matrix, reading n source elements and writing k.
     * @tparam WorkType element type of the optional scratch buffer (default T)
     * @tparam F kernel functor template, called as F()(src, dst, n, k, work)
     * @param m input matrix
     * @param n number of source elements the kernel consumes
     * @param k number of outputs the kernel writes
     * @param work_size if > 0, a WorkType[work_size] scratch buffer is passed as work
     * @param keep_shape true: keep m's shape; false: reshape result to {k}
     * @return result matrix
     */
    template<typename T, typename TargetType, typename WorkType = T, template<typename, typename, typename> class F>
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
            work = unique_ptr<WorkType[]>(new WorkType[work_size]);
        }
        F<decltype(m.getData()), decltype(r.getData()), decltype(work.get())>()(m.getData(), r.getData(), n, k, work.get());
        return r;
    }

    /**
     * Reduce along dimension dim_idx with a binary fold fun(acc, x), collapsing that axis
     * to length 1 (e.g. sum or product along an axis).
     * @param m input matrix
     * @param fun_1d binary fold, fun(const TargetType& acc, const T& x) -> new accumulator
     * @param dim_idx dimension reduced over (becomes length 1)
     * @param init_value initial accumulator value for each line
     * @return result matrix (dim_idx collapsed to length 1)
     */
    template<typename T, typename TargetType, typename F, typename Init = int>
    auto apply_function_dim_reduce(const Matrix<T> &m, F fun_1d, const msize_type &dim_idx,
                                   const Init &init_value = 0)
            -> Matrix<TargetType> {
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

    /**
     * Size-preserving map along dimension dim_idx: result element = fun(result, source),
     * seeded with init_value; shape unchanged. For a stateless fun this equals
     * apply_function_global_map.
     * @param m input matrix
     * @param fun binary map, fun(const TargetType& acc, const T& x) -> result element
     * @param dim_idx dimension that orders the traversal
     * @param init_value initial value of each result element
     * @return result matrix (same shape as m)
     */
    template<typename T, typename TargetType, typename F>
    auto apply_function_dim_map(const Matrix<T> &m, F fun, const msize_type &dim_idx,
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
                auto inner_iter_r = Internal::makeDimIter(iter_r + i, step_size, step_num);
                auto inner_iter_m = Internal::makeDimIter(iter_m + i, step_size, step_num);
                for (msize_type j = 0; j < step_num; ++j) {
                    auto &r_v = inner_iter_r[j];
                    auto &m_v = inner_iter_m[j];
                    r_v = fun(r_v, m_v);
                }
            }
            iter_m += unit_size;
            iter_r += unit_size;
        }
        return r;
    }

    /**
     * Like apply_function_dim_map, but the callback also gets j, the element's 0-based
     * index along dim_idx. Used by per-axis cumulative ops (cumsum/cumprod), which reset
     * at j == 0.
     * @param m input matrix
     * @param fun_index binary map with index, fun(const TargetType& acc, const T& x, size_type j)
     * @param dim_idx dimension whose position is passed as j
     * @param init_value initial value of each result element
     * @return result matrix (same shape as m)
     */
    template<typename T, typename TargetType, typename F>
    auto apply_function_dim_map_idx(const Matrix<T> &m, F fun_index, const msize_type &dim_idx,
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
                auto inner_iter_r = Internal::makeDimIter(iter_r + i, step_size, step_num);
                auto inner_iter_m = Internal::makeDimIter(iter_m + i, step_size, step_num);
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

    /**
     * Apply a 1-D kernel along dimension dim_idx, mapping each line to k outputs so that
     * axis becomes length k. Per line: fun_1d(src_iter, dst_iter, k, work).
     * Returns an empty matrix if k <= 0.
     * @tparam WorkType element type of the optional scratch buffer (default T)
     * @param m input matrix
     * @param fun_1d kernel, void(SrcDimIter src, DstDimIter dst, msize_type k, WorkType* work)
     * @param dim_idx dimension processed line-by-line (becomes length k)
     * @param k number of outputs per line
     * @param work_size if > 0, a WorkType[work_size] scratch buffer is passed as work
     * @return result matrix (dim_idx set to length k)
     */
    template<typename T, typename TargetType, typename WorkType = T, typename F>
    auto apply_function_dim_k(const Matrix<T> &m, F fun_1d, const msize_type &dim_idx, const msize_type &k,
                              const size_t &work_size = 0) -> Matrix<TargetType> {
        if (k <= 0) {
            throw_warning("For apply_function_dim_k, the k should be greater than 0, but got k = " + to_string(k) + ". Return an empty matrix.");
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
                auto inner_iter_m = Internal::makeDimIter(iter_m + i, step_size, step_num);
                auto inner_iter_r = Internal::makeDimIter(iter_r + i, step_size, k);
                fun_1d(inner_iter_m, inner_iter_r, k, work.get());
            }
            iter_m += unit_size;
            iter_r += r_unit_size;
        }
        return r;
    }

    /**
     * Functor-type variant of apply_function_dim_k: the per-line kernel is a functor
     * template F, invoked as F()(src_iter, dst_iter, k, work), mapping each line along
     * dim_idx to k outputs. Returns an empty matrix if k <= 0.
     * @tparam WorkType element type of the optional scratch buffer (default T)
     * @tparam F kernel functor template, called as F()(src, dst, k, work)
     * @param m input matrix
     * @param dim_idx dimension processed line-by-line (becomes length k)
     * @param k number of outputs per line
     * @param work_size if > 0, a WorkType[work_size] scratch buffer is passed as work
     * @return result matrix (dim_idx set to length k)
     */
    template<typename T, typename TargetType, typename WorkType = T, template<typename, typename, typename> class F>
    auto apply_tfunction_dim_k(const Matrix<T> &m, const msize_type &dim_idx, const msize_type &k,
                               const size_t &work_size = 0) -> Matrix<TargetType> {
        if (k <= 0) {
            throw_warning("For apply_tfunction_dim_k, the k should be greater than 0, but got k = " + to_string(k) + ". Return an empty matrix.");
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
                auto inner_iter_m = Internal::makeDimIter(iter_m + i, step_size, step_num);
                auto inner_iter_r = Internal::makeDimIter(iter_r + i, step_size, k);
                F<decltype(inner_iter_m), decltype(inner_iter_r), decltype(work.get())>()(inner_iter_m, inner_iter_r, k,
                                                                                          work.get());
            }
            iter_m += unit_size;
            iter_r += r_unit_size;
        }
        return r;
    }

    /**
     * Functor-type kernel along dimension dim_idx with explicit input length:
     * F()(src_iter, dst_iter, n, k, work) maps each line's n elements to k outputs so that
     * axis becomes length k. Returns an empty matrix if k <= 0.
     * @tparam WorkType element type of the optional scratch buffer (default T)
     * @tparam F kernel functor template, called as F()(src, dst, n, k, work)
     * @param m input matrix
     * @param dim_idx dimension processed line-by-line (becomes length k)
     * @param n number of source elements read per line
     * @param k number of outputs per line
     * @param work_size if > 0, a WorkType[work_size] scratch buffer is passed as work
     * @return result matrix (dim_idx set to length k)
     */
    template<typename T, typename TargetType, typename WorkType = T, template<typename, typename, typename> class F>
    auto apply_tfunction_dim_n_k(const Matrix<T> &m, const msize_type &dim_idx, const msize_type &n, const msize_type &k,
                                 const size_t &work_size = 0) -> Matrix<TargetType> {
        if (k <= 0) {
            throw_warning("For apply_tfunction_dim_n_k, the k should be greater than 0, but got k = " + to_string(k) + ". Return an empty matrix.");
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
                auto inner_iter_m = Internal::makeDimIter(iter_m + i, step_size, n);
                auto inner_iter_r = Internal::makeDimIter(iter_r + i, step_size, k);
                F<decltype(inner_iter_m), decltype(inner_iter_r), decltype(work.get())>()(inner_iter_m, inner_iter_r, n, k,
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
