// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2022/7/2.
//
#ifndef MATSAR_MATRIX_HPP
#pragma once
#define MATSAR_MATRIX_HPP

#include <cstddef>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <cstdarg>
#include <iomanip>

#ifndef MATSAR_UTILS

#include "util/Utils"

#endif

#ifndef MATSAR_NUMBEROPERATION_HPP

#include "NumberOperation.hpp"

#endif

#define MATRIX_SIZE_TYPE int64
using msize_type = MATRIX_SIZE_TYPE;
using mshape_type = std::vector<msize_type>;

#include <complex>

#ifdef MATSAR_WITH_BLAS

#include <cblas.h>

// NOTE: <lapacke.h> and its lapack_* configuration macros are intentionally NOT
// included here. The core (mmul / operator%) uses only CBLAS, which every BLAS
// implementation provides. LAPACKE is required solely by the experimental
// linear-algebra module, which now includes <lapacke.h> itself (see
// math/LinearAlgebra.hpp). Keeping it out of the core lets the core build with
// any OpenBLAS on every platform -- notably Debian/Ubuntu, where lapacke.h ships
// in a separate liblapacke-dev package and is NOT exposed by OpenBLAS's
// pkg-config include path.

#endif // MATSAR_WITH_BLAS

// Hand-laid-out code-generation macros below; keep their exact layout.
// clang-format off
#define LIST_CONSTRUCTOR_MAKER(LIST_n) \
Matrix(LIST_n list) {\
    Util::VectorUtil::checkListFix(list);\
    this->size = Util::VectorUtil::getFixListSize(list); \
    this->shape = Util::VectorUtil::getDimSize(list);\
    allocate(this->size);\
    Util::VectorUtil::uninitListCopy(list, this->data);\
}

#define LIST_CONSTRUCTOR_MAKER_N \
LIST_CONSTRUCTOR_MAKER(LIST##3)\
LIST_CONSTRUCTOR_MAKER(LIST##4)\
LIST_CONSTRUCTOR_MAKER(LIST##5)\
LIST_CONSTRUCTOR_MAKER(LIST##6)\
LIST_CONSTRUCTOR_MAKER(LIST##7)\
LIST_CONSTRUCTOR_MAKER(LIST##8)


#define VEC_CONSTRUCTOR_MAKER(VEC_n) \
Matrix(VEC_n vec) {\
    Util::VectorUtil::checkVecFix(vec);\
    this->size = Util::VectorUtil::getFixVecSize(vec); \
    this->shape = Util::VectorUtil::getVecDimSize(vec);\
    allocate(this->size);\
    Util::VectorUtil::VecCopy(vec, this->data);\
}

#define VEC_CONSTRUCTOR_MAKER_N \
VEC_CONSTRUCTOR_MAKER(VEC##3)\
VEC_CONSTRUCTOR_MAKER(VEC##4)\
VEC_CONSTRUCTOR_MAKER(VEC##5)\
VEC_CONSTRUCTOR_MAKER(VEC##6)\
VEC_CONSTRUCTOR_MAKER(VEC##7)\
VEC_CONSTRUCTOR_MAKER(VEC##8)
// clang-format on

using namespace matsar::Util::TypeUtil;
using matsar::Util::TypeUtil::getCalculableString;
using matsar::Util::TypeUtil::typeHigher;

using namespace matsar::Util::VectorUtil;
using matsar::Util::VectorUtil::mul;
using matsar::Util::VectorUtil::toString;
using matsar::Util::VectorUtil::to_string;

using namespace matsar::Util::Exception;

namespace matsar {
    template<typename dType>
    void swap(dType &left, dType &right) noexcept {
        auto tmp = left;
        left = right;
        right = tmp;
    }

    /// Internal, element-type-erased base for Matrix: shape/broadcast math and the
    /// low-level kernels. The `///NO CHECK!` helpers are unchecked and not part of
    /// the public API -- use Matrix<dType> directly.
    class MatrixBase {
    protected:
        using size_type = msize_type;

    public:
        static void checkShape(const mshape_type &shape) {
            for (const auto &i: shape) {
                if (i < 0) {
                    throw_value_error("Matrix shape cannot be less than 0, shape = " + toString(shape));
                }
            }
            if (shape.size() > std::numeric_limits<msize_type>::max()) {
                throw_shape_exception(
                        "Matrix shape dimension size cannot be greater than INT64_MAX, shape = " + toString(shape));
            }
        }

        static auto newAxis(const mshape_type &shape) -> mshape_type {
            checkShape(shape);
            if (shape.empty())
                return {};
            mshape_type new_shape(shape);
            new_shape.push_back(1);
            return new_shape;
        }

        static auto newAxis(const mshape_type &shape, const size_type dim) -> mshape_type {
            checkShape(shape);
            if (shape.empty())
                return {};
            if (shape.size() > dim)
                return shape;
            mshape_type new_shape(shape);
            alignNum(new_shape, dim, static_cast<size_type>(1));
            return new_shape;
        }

        static auto squeezeShape(const mshape_type &shape) -> mshape_type {
            checkShape(shape);
            mshape_type squeezed_shape;
            for (auto i: shape)
                if (i != 1)
                    squeezed_shape.push_back(i);
            if (squeezed_shape.empty())
                squeezed_shape.push_back(1);
            return squeezed_shape;
        }

        /**
         * Only remove the 1s of high dimension of shape
         */
        static auto squeezeShapeHigh(const mshape_type &shape) -> mshape_type {
            checkShape(shape);
            mshape_type squeezed_shape;
            msize_type idx = 0;
            for (msize_type i = 0; i < shape.size(); ++i)
                if (shape[shape.size() - i - 1] != 1) {
                    idx = static_cast<msize_type>(shape.size()) - i - 1;
                    break;
                }
            for (msize_type i = 0; i <= idx; ++i)
                squeezed_shape.push_back(shape[i]);
            if (squeezed_shape.empty())
                squeezed_shape.push_back(1);
            return squeezed_shape;
        }

        static auto broadcastShape(mshape_type shape1, mshape_type shape2, const msize_type dim = 1) -> mshape_type {
            checkShape(shape1);
            checkShape(shape2);
            const auto dim_idx = dim - 1;
            const msize_type shape_dim = static_cast<msize_type>(max(shape1.size(), shape2.size()));
            alignNum(shape1, shape_dim, static_cast<msize_type>(1));
            alignNum(shape2, shape_dim, static_cast<msize_type>(1));
            mshape_type shape(shape_dim);
            for (msize_type i = dim_idx; i < shape1.size(); ++i)
                if (shape1[i] == shape2[i] || shape2[i] == 1)
                    shape[i] = shape1[i];
                else if (shape1[i] == 1)
                    shape[i] = shape2[i];
                else
                    throw_broadcast_exception(toString(shape1), toString(shape2));
            return shape;
        }

        static void shapePredict(size_type target_size, mshape_type &shape) {
            auto iter = std::find(shape.begin(), shape.end(), -1);
            if (iter != shape.end()) {
                if (std::find(iter + 1, shape.end(), -1) != shape.end())//find second -1
                    throw_value_error("Matrix dimensions can only declare one unknown dimension (-1)");
                auto predict_dim_size = target_size;
                for (const auto &i: shape) {
                    if (i == -1)
                        continue;
                    if (predict_dim_size % i != 0)
                        throw_value_error(
                                "Cannot be refactored shape size=" + std::to_string(target_size) + "to shape=" +
                                matsar::Util::VectorUtil::toString(shape));
                    predict_dim_size /= i;
                }
                *iter = static_cast<size_type>(predict_dim_size);
            }
        };

        static auto shapeToString(const mshape_type &shape, const string &segment = "x") -> string {
            stringstream ss;
            ss << "(shape:";
            ss << Util::VectorUtil::toString(shape, segment);
            ss << ")";
            return ss.str();
        }

        template<typename Ts, typename Td>
        static void T(Ts *src, Td *dst, size_type dim1, size_type dim2) {
            for (size_type i = 0; i < dim1; ++i)
                for (size_type j = 0; j < dim2; ++j)
                    *(dst++) = src[i + j * dim1];
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Add(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ += static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Add(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ += tmp;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Sub(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ -= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Sub(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ -= tmp;
        }

#ifdef MATSAR_WITH_BLAS
        /// Narrowing guard: BLAS dimensions must fit the library's native blasint.
        static blasint to_blasint(size_type v) {
            if (v < 0 || v > static_cast<size_type>(std::numeric_limits<blasint>::max()))
                throw_value_error("matrix dimension exceeds BLAS index range");
            return static_cast<blasint>(v);
        }

        ///NO CHECK!
        static void MMul(float32 *left, float32 *right, float32 *dst, size_type m, size_type n, size_type k,
                         size_type unit_times) {
            const blasint bm = to_blasint(m), bn = to_blasint(n), bk = to_blasint(k);
            while (unit_times-- > 0) {
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            bm, bn, bk, 1, left, bm, right, bk, 0, dst, bm);
                left += m * k;
                right += k * n;
                dst += m * n;
            }
        }

        ///NO CHECK!
        static void MMul(float64 *left, float64 *right, float64 *dst, size_type m, size_type n, size_type k,
                         size_type unit_times) {
            const blasint bm = to_blasint(m), bn = to_blasint(n), bk = to_blasint(k);
            while (unit_times-- > 0) {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            bm, bn, bk, 1., left, bm, right, bk, 0, dst, bm);
                left += m * k;
                right += k * n;
                dst += m * n;
            }
        }

        ///NO CHECK!
        static void MMul(complex64 *left, complex64 *right, complex64 *dst, size_type m, size_type n, size_type k,
                         size_type unit_times) {
            complex64 one(1);
            complex64 zero;
            const blasint bm = to_blasint(m), bn = to_blasint(n), bk = to_blasint(k);
            while (unit_times-- > 0) {
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            bm, bn, bk, &one, left, bm, right, bk, &zero, dst, bm);
                left += m * k;
                right += k * n;
                dst += m * n;
            }
        }

        ///NO CHECK!
        static void MMul(complex128 *left, complex128 *right, complex128 *dst, size_type m, size_type n, size_type k,
                         size_type unit_times) {
            complex128 one(1);
            complex128 zero;
            const blasint bm = to_blasint(m), bn = to_blasint(n), bk = to_blasint(k);
            while (unit_times-- > 0) {
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            bm, bn, bk, &one, left, bm, right, bk, &zero, dst, bm);
                left += m * k;
                right += k * n;
                dst += m * n;
            }
        }

#endif // MATSAR_WITH_BLAS

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Mul(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ *= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Mul(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ *= tmp;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Div(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ /= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Div(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ /= tmp;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Div(const leftT &left, rightT *right, size_type n) {
            auto tmp = static_cast<rightT>(left);
            while (n-- > 0) {
                *right = tmp / *right;
                ++right;
            }
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Rem(leftT *left, rightT *right, size_type n) {
            while (n-- > 0) {
                *left = rem(*left, static_cast<leftT>(*right));
                ++left, ++right;
            }
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Rem(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0) {
                *left = rem(*left, tmp);
                ++left;
            }
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Mod(leftT *left, rightT *right, size_type n) {
            while (n-- > 0) {
                *left = mod(*left, static_cast<leftT>(*right));
                ++left, ++right;
            }
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Mod(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0) {
                *left = mod(*left, tmp);
                ++left;
            }
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void And(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ &= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void And(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ &= tmp;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Or(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ |= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Or(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ |= tmp;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Xor(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ ^= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<typename leftT, typename rightT>
        static void Xor(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ ^= tmp;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Less(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ < *right++;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Less(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ < right;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void LessEqual(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ <= *right++;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void LessEqual(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ <= right;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Great(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ > *right++;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Great(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ > right;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void GreatEqual(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ >= *right++;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void GreatEqual(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ >= right;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Equal(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ == *right++;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Equal(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ == right;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void NotEqual(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ != *right++;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void NotEqual(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ != right;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Pow(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = std::pow(*left++, *right++);
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Pow(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = std::pow(*left++, right);
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Pow(const leftT &left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = std::pow(left, *right++);
        }
    };

    /**
     * N-dimensional, header-only matrix container templated on the element type
     * `dType` (e.g. int64, float64, complex128).
     *
     * Construction from nested initializer lists is ROW-wise by default
     * (`{{1,2,3},{4,5,6}}` is 2x3); `fromRows`/`fromColumns` build explicitly.
     * `operator()` and the `toString`/`toRowColString` views are row-wise too.
     *
     * Internal storage is COLUMN-major: iteration (`begin`/`end`, range-for),
     * `getData()`, and `reshape()` follow storage order, which does NOT match the
     * row-wise reading. See docs/usage.md for the full layout note.
     *
     * Arithmetic and comparisons (`+ - * /`, `< <= > >= == !=`) are element-wise
     * with broadcasting. Note `*` is element-wise -- the matrix product is
     * `mmul(a, b)` / `a % b`.
     */
    template<typename dType>
    class Matrix : public MatrixBase {
    public:
        using size_type = msize_type;
        using shape_type = mshape_type;

    private:
        using alloc_trait = allocator_traits<allocator<dType>>;

    protected:
        using Base = MatrixBase;
        allocator<dType> alloc;
        dType *data = nullptr;
        dType *last = nullptr;
        size_type size = 0;
        shape_type shape = {0};

        /**
         * Automatic data assignment, data and last memory management
         */
        auto allocate(size_type n, const void *hint = nullptr) -> dType * {
            try {
                if (hint)
                    data = alloc_trait::allocate(alloc, n, hint);
                else
                    data = alloc_trait::allocate(alloc, n);
                last = data + n;
                return data;
            } catch (const bad_alloc &e) {
                cerr << e.what() << ": " << "Unable to allocate" + to_string(n * sizeof(dType)) + "B memory for an array with shape " + matsar::Util::VectorUtil::toString(shape) + " and data type "
                     << typeid(dType).name() << endl;
                debug();
            } catch (const exception &e) {
                cerr << e.what() << ": " << "Unable to allocate but not bad_alloc" << endl;
                debug();
            }
            data = last = nullptr;
            return nullptr;
        }

        /**
         * Automatic data assignment, data and last memory management
         */
        void deallocate() {
            try {
                if (data)
                    alloc_trait::deallocate(alloc, data, size);
                data = last = nullptr;
            } catch (const exception &e) {
                cerr << e.what() << ": " << "Deallocate Failed!" << endl;
                debug();
            }
        }

        static auto dataToString(bool p) -> string {
            return p ? "true " : "false";
        }

        template<typename RT>
        static auto dataToString(RT p, int dataSpace = 10) -> string {
            stringstream ss;
            ss.setf(ios::fixed);
            if (p == RT(0))
                ss.precision(0);
            else
                ss.precision(4);
            ss << setfill(' ') << setw(dataSpace) << p;
            return ss.str();
        }

        template<typename T>
        static auto dataToString(complex<T> p, int dataSpace = 10) -> string {
            stringstream ss;
            ss.setf(ios::fixed);
            ss.precision(4);
            ss << setfill(' ') << setw(dataSpace) << p.real();
            if (p.imag() < 0)
                ss << " - ";
            else
                ss << " + ";
            ss << setfill(' ') << setw(dataSpace) << std::abs(p.imag()) << "i";
            return ss.str();
        }

        /// Logical (row-major) braced rendering: outermost brace = axis 0, innermost =
        /// last axis, elements accessed by column-major offset so the printout matches
        /// operator()/toRowColString. `base` is the storage offset of the current slice.
        auto arrayToString(msize_type dim, size_type base, int align = 0) const -> string {
            if (!data || shape.empty() || dim < 0 || dim >= static_cast<msize_type>(shape.size()))
                return "{}";
            stringstream ss;
            string spaces(static_cast<std::basic_string<char>::size_type>((align + 1) * 2), ' ');
            // column-major stride of axis `dim` = product(shape[0..dim-1])
            size_type stride = 1;
            for (msize_type d = 0; d < dim; ++d)
                stride *= shape[d];
            if (dim == static_cast<msize_type>(shape.size()) - 1) {
                ss << spaces << "{";
                for (size_type k = 0; k < shape[dim]; ++k) {
                    ss << dataToString(data[base + k * stride]) << ' ';
                    if (k + 1 == shape[dim])
                        ss << "}";
                    else
                        ss << ", ";
                }
                return ss.str();
            }
            ss << spaces << "{\n";
            for (size_type i = 0; i < shape[dim]; ++i) {
                ss << arrayToString(dim + 1, base + i * stride, align + 1);
                if (i != shape[dim] - 1) {
                    ss << ",\n";
                    if (dim == static_cast<msize_type>(shape.size()) - 3)
                        ss << "\n";
                }
            }
            ss << "\n"
               << spaces << "}";
            return ss.str();
        }

        /**
         * single unit 2 dim matrix get dType& element
         * NO CHECK!
         */
        [[nodiscard]] auto getEle(dType *src, size_type i, size_type j) const -> dType & {
            return src[i + j * shape[0]];
        }

        /**
         * single unit 2 dim matrix get dType& element
         * NO CHECK!
         */
        [[nodiscard]] auto getEle(size_type n, size_type i, size_type j) const -> dType & {
            return (data + n * shape[0] * shape[1])[i + j * shape[0]];
        }

        /**
         * single unit 2 dim matrix get dType& element
         * NO CHECK!
         */
        [[nodiscard]] auto getEle(size_type i, size_type j) const -> dType & {
            return (data)[i + j * shape[0]];
        }

        /**
         * return a new Matrix for only 2d matrix \n
         * only works when row col all valid \n
         * @Hints: Do Det Operations can be easy to just implement the actual data removal operations. \n
         * But this way can be massively memory required. \n
         * There is better way to just overlook the removed columns and rows and do calculations.
         */
        auto remove_row_col(dType *src, size_type row, size_type col) const -> Matrix<dType> {
            if (row < 0 || col < 0 || row >= shape[0] || col >= shape[1])
                return {{shape[0], shape[1]}, src};
            if (shape[0] == 1 || shape[1] == 1)
                return {};
            shape_type removed_shape = {shape[0] - 1, shape[1] - 1};
            Matrix<dType> m(removed_shape, 0.);
            dType *iter_m = m.getData();
            for (size_type j = 0; j < shape[1]; ++j) {
                if (j == col) continue;
                for (size_type i = 0; i < shape[0]; ++i) {
                    if (i == row) continue;
                    *(iter_m++) = getEle(src, i, j);
                }
            }
            return m;
        }

        /**
         * single unit 2 dim matrix \n
         * This function gets the 2-dimensional det of a page by moving src \n
         * Potential Flaw: Too slow, Stack Overflow and Memory Overflow \n
         * @Hints: Do Det Operations can be easy to just implement the actual data removal operations. \n
         * But this way can be massively memory required. \n
         * There is better way to just overlook the removed columns and rows and do calculations.
         */
        auto Det(dType *src) const -> dType {
            if (shape[0] == 1)
                return *src;
            size_type i = 0;
            dType ans = 0;
            for (size_type j = 0; j < shape[0]; ++j) {
                auto num = getEle(src, i, j);
                if (num == 0) continue;
                if ((i + j) % 2 == 0)
                    ans += num * remove_row_col(src, i, j).det();
                else
                    ans -= num * remove_row_col(src, i, j).det();
            }
            return ans;
        }

        /**
         * single unit 2 dim matrix
         * require element at least float
         * force type tran is in public inv
         */
        void inv(dType *dst) const {
            if (shape[0] != shape[1]) {
                throw_value_error("inv only for square matrix");
                return;
            }
            auto n = shape[0];
            unique_ptr<size_type[]> is(new size_type[n]);
            unique_ptr<size_type[]> js(new size_type[n]);
            size_type i, j, k;
            for (k = 0; k < n; ++k) {
                dType max = 0;
                for (i = k; i < n; ++i)
                    for (j = k; j < n; ++j) {
                        dType tmp = std::abs(getEle(dst, i, j));
                        if (tmp > max) {
                            max = tmp;
                            is[k] = i;
                            js[k] = j;
                        }
                    }
                if (max == 0.) {
                    throw_value_error("This matrix does not have an inverse matrix");
                    return;
                }
                if ((i = is[k]) != k)
                    for (j = 0; j < n; ++j)
                        swap(*(dst + k * n + j), *(dst + i * n + j));
                if ((j = js[k]) != k)
                    for (i = 0; i < n; i++)
                        swap(*(dst + i * n + k), *(dst + i * n + j));
                getEle(dst, k, k) = 1. / getEle(dst, k, k);
                for (j = 0; j < n; j++)
                    if (j != k)
                        getEle(dst, k, j) *= getEle(dst, k, k);
                for (i = 0; i < n; i++)
                    if (i != k)
                        for (j = 0; j < n; j++)
                            if (j != k)
                                getEle(dst, i, j) =
                                        getEle(dst, i, j) - getEle(dst, i, k) * getEle(dst, k, j);
                for (i = 0; i < n; i++)
                    if (i != k)
                        getEle(dst, i, k) *= -getEle(dst, k, k);
            }
            for (k = n - 1; k >= 0; k--) {
                if ((j = js[k]) != k)
                    for (i = 0; i < n; i++)
                        swap(*(dst + j * n + i), *(dst + k * n + i));
                if ((i = is[k]) != k)
                    for (j = 0; j < n; j++)
                        swap(*(dst + j * n + i), *(dst + j * n + k));
            }
        }

        template<typename T>
        static auto cat(size_type dim, vector<Matrix<T>> mv) -> Matrix<T> {
            if (dim < 0) throw_value_error("The dim parameter must be a positive real number.", true);
            if (mv.empty()) return {};
            for (auto &m: mv) {
                m.squeezeHighThis();
                m.expandShape(dim);
            }
            auto dim_idx = dim - 1;
            auto r_shape = mv[0].getShape();
            r_shape.erase(r_shape.begin() + dim_idx);
            size_type dim_sum = 0;
            for (const auto &m: mv) {
                auto shape = m.getShape();
                dim_sum += shape[dim_idx];
                shape.erase(shape.begin() + dim_idx);
                if (shape != r_shape)
                    throw_value_error("cat matrix dimensions do not match", true);
            }
            r_shape.insert(r_shape.begin() + dim_idx, dim_sum);
            Matrix<T> r(r_shape, 0.);
            if (dim == 1) {
                size_type dim_offset = 0;
                for (size_type mi = 0; mi < mv.size(); ++mi) {
                    shape_type selector(r_shape.size(), 0);
                    auto current_shape = mv[mi].getShape();
                    auto size = mv[mi].getSize();
                    while (size-- > 0) {
                        auto off_selector = selector;
                        off_selector[dim_idx] += dim_offset;
                        r(off_selector) = mv[mi](selector);
                        for (size_type i = 0; i < selector.size(); ++i) {
                            if (++selector[i] == current_shape[i])
                                selector[i] = 0;
                            else
                                break;
                        }
                    }
                    dim_offset += current_shape[dim_idx];
                }
            } else {
                auto iter_r = r.getData();
                for (size_type mi = 0; mi < mv.size(); ++mi) {
                    uninitialized_copy_n(mv[mi].getData(), mv[mi].getSize(), iter_r);
                    iter_r += mv[mi].getSize();
                }
            }
            return r;
        }

    public:
        using dataType = dType;

        void debug() const {
            stringstream ss;
            ss << "Matrix debug message: "
               << " shape = " << matsar::Util::VectorUtil::toString(shape)
               << " size = " << size
               << " data = " << data
               << " matrix * = " << this;
            cerr << ss.str() << endl;
        }

        [[nodiscard]] virtual auto getData() const -> dType * {
            return data;
        }

        [[nodiscard]] virtual auto begin() const -> dType * {
            return data;
        }

        [[nodiscard]] auto getDataEnd() const -> dType * {
            return last;
        }

        [[nodiscard]] virtual auto end() const -> dType * {
            return last;
        }

        [[nodiscard]] auto getSize() const -> size_type {
            return size;
        }

        [[nodiscard]] auto getShape() const -> shape_type {
            return shape;
        }

        [[nodiscard]] auto getDims() const -> size_type {
            return shape.size();
        }

        /**
         * The length of the maximum dimension of the matrix
         */
        [[nodiscard]] auto getLength() const -> size_type {
            return Util::VectorUtil::max(shape);
        }

        ///unit-Two-dimensional matrix unit
        [[nodiscard]] auto getUnitSize() const -> size_type {
            switch (getDims()) {
            case 0:
                return 0;
            case 1:
                return shape[0];
            default:
            case 2:
                return shape[0] * shape[1];
            }
        }

        ///unit-Two-dimensional matrix unit
        [[nodiscard]] auto getUnitNum() const -> size_type {
            if (size == 0)
                return 0;
            return size / getUnitSize();
        }

        ///unit-Two-dimensional matrix unit
        [[nodiscard]] auto getUnitShape() const -> shape_type {
            switch (getDims()) {
            case 0:
                return {0, 0};
            case 1:
                return {shape[0], 1};
            default:
            case 2:
                return {shape[0], shape[1]};
            }
        }

        void setShape(shape_type new_shape) {
            shapePredict(size, new_shape);
            if (Util::VectorUtil::mul(new_shape) != size)
                throw_value_error("reshape new shape must have the same size of matrix shape:" +
                                  Util::VectorUtil::toString(shape) + " new shape=" +
                                  Util::VectorUtil::toString(new_shape));
            shape = new_shape;
        }

        template<typename TargetType = decltype(std::real(std::declval<dType>()))>
        [[nodiscard]] auto real() const -> Matrix<TargetType> {
            Matrix<TargetType> r(shape, TargetType());
            auto iter_c = getData();
            for (auto *iter = r.getData(), *end = r.getDataEnd(); iter != end; ++iter, ++iter_c)
                *iter = matsar::real(*iter_c);
            return r;
        }

        template<typename TargetType = decltype(std::imag(std::declval<dType>()))>
        [[nodiscard]] auto image() const -> Matrix<TargetType> {
            Matrix<TargetType> r(shape, TargetType());
            auto iter_c = getData();
            for (auto *iter = r.getData(), *end = r.getDataEnd(); iter != end; ++iter, ++iter_c)
                *iter = matsar::imag(*iter_c);
            return r;
        }

        template<typename TargetType = decltype(std::imag(std::declval<dType>()))>
        [[nodiscard]] auto imag() const -> Matrix<TargetType> {
            return this->image<TargetType>();
        }

        template<typename TargetType = decltype(std::conj(std::declval<dType>()))>
        [[nodiscard]] auto conj() const -> Matrix<TargetType> {
            Matrix<TargetType> r(shape, TargetType());
            auto iter_c = getData();
            for (auto *iter = r.getData(), *end = r.getDataEnd(); iter != end; ++iter, ++iter_c) {
                *iter = matsar::conj(*iter_c);
            }
            return r;
        }

        ///DTYPE TO DTYPE
        auto operator=(const Matrix<dType> &m) -> Matrix<dType> & {
            if (&m == this) return *this;//handle same!!!
            if (this->size != m.size) {
                this->deallocate();
                allocate(m.getSize(), m.getData());
            }
            uninitialized_copy_n(m.getData(), m.getSize(), this->data);
            this->size = m.getSize();
            this->shape = m.getShape();
            return *this;
        }

        template<typename T>
        auto operator=(const Matrix<T> &m) -> Matrix<dType> & {
            this->operator=(Matrix<dType>(m));
            return *this;
        }

        auto operator=(const dType &m) -> Matrix<dType> & {
            this->operator=(Matrix<dType>({m}));
            return *this;
        }

        template<Calculable T CheckCalculable(T)>
        auto operator=(const T &m) -> Matrix<dType> & {
            if (typeHigher<T, dType>())
                throw_type_loss_exception<T, dType>(false);
            this->operator=(dType(m));
            return *this;
        }

        Matrix() = default;

        ///DEFAULT ANOTHER
        Matrix(const Matrix &another) {
            if (!another.isEmpty()) {
                allocate(another.size, another.data);
                uninitialized_copy_n(another.data, another.size, this->data);
                this->size = another.size;
                this->shape = another.shape;
            }
        };

        template<typename T>
        Matrix(const Matrix<T> &another) {
            if (typeHigher<T, dType>())
                throw_type_loss_exception<T, dType>(false);
            if (!another.isEmpty()) {
                allocate(another.getSize(), another.getData());
                uninitialized_copy_n(another.getData(), another.getSize(), this->data);
                this->size = another.getSize();
                this->shape = another.getShape();
            }
        };

        template<typename T>
        Matrix(const vector<Matrix<T>> &ms, size_type dim = 1) {
            if (typeHigher<T, dType>())
                throw_type_loss_exception<T, dType>(false);
            this->operator=(cat(dim, vector<Matrix<dType>>(ms)));
        };

        template<typename T>
        Matrix(const initializer_list<Matrix<T>> &ms, size_type dim = 1) {
            if (typeHigher<T, dType>())
                throw_type_loss_exception<T, dType>(false);
            vector<Matrix<dType>> mv(ms.begin(), ms.end());
            this->operator=(cat(dim, mv));
        };

        explicit Matrix(dType v) {
            this->shape = {1};
            this->size = 1;
            allocate(size);
            uninitialized_fill_n(data, size, v);
        };

        explicit operator dType() {
            return *data;
        }

        /**
         * SINGLE VALUE GENERATOR
         * don't set default value for v because there are signatures for direct vector input construction
         */
        explicit Matrix(const shape_type &m_shape, const dType &v, const void *hint = nullptr) {
            Base::checkShape(m_shape);
            this->shape = Util::VectorUtil::abs(m_shape);
            this->size = Util::VectorUtil::mul(this->shape);
            if (size == 0) {
                this->shape = {0};
            } else {
                allocate(size, hint);
                uninitialized_fill_n(data, size, v);
            }
        };

        Matrix(const shape_type &m_shape, dType *v) {
            Base::checkShape(m_shape);
            this->shape = Util::VectorUtil::abs(m_shape);
            this->size = Util::VectorUtil::mul(this->shape);
            if (size == 0) {
                this->shape = {0};
            } else {
                allocate(size, v);
                if (v)
                    uninitialized_copy_n(v, size, data);
                else
                    uninitialized_fill_n(data, size, 0);
            }
        };

        ///LIST GENERATOR
        Matrix(LIST1 list) {
            allocate(list.size());
            uninitialized_copy(list.begin(), list.end(), this->data);
            this->size = list.size();
            this->shape = {this->size};
        }

        /// Build a matrix from explicit COLUMNS: each inner list is one column
        /// (this is the legacy nested-initializer-list layout). Column-major stored.
        static auto fromColumns(LIST2 list) -> Matrix<dType> {
            Matrix<dType> m;
            Util::VectorUtil::checkListFix(list);
            m.size = Util::VectorUtil::getFixListSize(list);
            m.shape = Util::VectorUtil::getDimSize(list);
            m.allocate(m.size);
            Util::VectorUtil::uninitListCopy(list, m.data);
            return m;
        }

        /// Build a matrix from explicit ROWS: each inner list is one row
        /// (NumPy/MATLAB-literal layout). Implemented as fromColumns(list).T().
        static auto fromRows(LIST2 list) -> Matrix<dType> {
            return fromColumns(list).T();
        }

        /// vector<vector> overloads of the explicit builders (same semantics as the
        /// initializer-list versions): fromColumns = column-wise, fromRows = transpose.
        static auto fromColumns(VEC2 vec) -> Matrix<dType> {
            Matrix<dType> m;
            Util::VectorUtil::checkVecFix(vec);
            m.size = Util::VectorUtil::getFixVecSize(vec);
            m.shape = Util::VectorUtil::getVecDimSize(vec);
            m.allocate(m.size);
            Util::VectorUtil::VecCopy(vec, m.data);
            return m;
        }

        static auto fromRows(VEC2 vec) -> Matrix<dType> {
            return fromColumns(vec).T();
        }

        /// 2-D nested initializer list. Default = row-wise (fromRows). Define
        /// MATSAR_LEGACY_COLUMN_INITIALIZER_LIST for legacy column-wise layout
        /// (temporary migration aid -- prefer fromRows()/fromColumns()).
        Matrix(LIST2 list) {
#ifdef MATSAR_LEGACY_COLUMN_INITIALIZER_LIST
            *this = fromColumns(list);
#else
            *this = fromRows(list);
#endif
        }

        LIST_CONSTRUCTOR_MAKER_N

        Matrix(VEC1 vec) {
            allocate(vec.size());
            uninitialized_copy(vec.begin(), vec.end(), this->data);
            this->size = vec.size();
            this->shape = {this->size};
        }

        /// 2-D nested vector. Default = row-wise (fromRows); legacy column-wise under
        /// MATSAR_LEGACY_COLUMN_INITIALIZER_LIST (consistent with the initializer-list ctor).
        Matrix(VEC2 vec) {
#ifdef MATSAR_LEGACY_COLUMN_INITIALIZER_LIST
            *this = fromColumns(vec);
#else
            *this = fromRows(vec);
#endif
        }

        VEC_CONSTRUCTOR_MAKER_N

        //TODO: implement constructor for different vector with different types

        ~Matrix() {
            deallocate();
        }

        [[nodiscard]] auto broadcast(shape_type target_shape) const -> Matrix<dType> {
            if (shape == target_shape) return {*this};
            Matrix<dType> sqz = squeezeHigh();
            Matrix<dType> r(target_shape, 0.);
            if (sqz.getDims() > target_shape.size())
                throw_broadcast_exception(MatrixBase::shapeToString(shape, ","),
                                          MatrixBase::shapeToString(target_shape, ","));

            alignNum(sqz.shape, target_shape.size(), (size_type) 1);
            dType *iter_r = r.data, *iter_sqz = sqz.data, *end_sqz = iter_sqz + sqz.size, *end_r;

            if (target_shape.empty()) throw_broadcast_exception("empty matrix broadcast fail!");
            size_type m_s = sqz.shape[0], n_s = sqz.getDims() > 1 ? sqz.shape[1] : 1;
            size_type m_t = target_shape[0], n_t = target_shape.size() > 1 ? target_shape[1] : 1;

            if (m_s == m_t && n_s == n_t) {
                //TODO all copy
                while (iter_sqz != end_sqz)
                    *(iter_r++) = *(iter_sqz++);
            } else if (m_s == m_t) {
                if (n_s != 1)
                    throw_broadcast_exception(MatrixBase::shapeToString(shape, ","),
                                              MatrixBase::shapeToString(target_shape, ","));
                //TODO copy vector
                for (size_type n = 0; n < getUnitNum(); ++n) {
                    for (size_type j = 0; j < n_t; ++j)
                        for (size_type i = 0; i < m_t; ++i)
                            *(iter_r++) = *(iter_sqz + i);// re-read from sqz
                    iter_sqz += sqz.getUnitSize();
                }
            } else if (n_s == n_t) {
                if (m_s != 1)
                    throw_broadcast_exception(MatrixBase::shapeToString(shape, ","),
                                              MatrixBase::shapeToString(target_shape, ","));
                for (size_type n = 0; n < getUnitNum(); ++n)
                    for (size_type j = 0; j < n_t; ++j) {
                        for (size_type i = 0; i < m_t; ++i)
                            *(iter_r++) = *iter_sqz;
                        iter_sqz++;
                    }
            } else {
                //TODO copy one to unit
                if (m_s != 1 || n_s != 1)
                    throw_broadcast_exception(MatrixBase::shapeToString(shape, ","),
                                              MatrixBase::shapeToString(target_shape, ","));
                size_type r_unit_size = r.getUnitSize();
                for (size_type n = 0; n < getUnitNum(); ++n) {
                    for (size_type i = 0; i < r_unit_size; ++i)
                        *(iter_r++) = *iter_sqz;
                    iter_sqz++;
                }
            }
            //TODO move copy
            for (size_type dim = 2; dim < r.getDims(); ++dim) {
                if (sqz.shape[dim] == target_shape[dim])
                    continue;
                if (sqz.shape[dim] != 1)
                    throw_broadcast_exception(MatrixBase::shapeToString(shape, ","),
                                              MatrixBase::shapeToString(target_shape, ","));
                end_r = iter_r;
                for (size_type i = 0; i < target_shape[dim] - 1; ++i) {//forward copy n times
                    dType *iter_r_tmp = r.data;
                    while (iter_r_tmp != end_r)
                        *(iter_r++) = *(iter_r_tmp++);
                }
            }
            return r;
        }

        template<typename T1, typename T2>
        static void broadcast(Matrix<T1> &m1, Matrix<T2> &m2) {
            if (m1.getShape() == m2.getShape())
                return;
            auto bShape = broadcastShape(m1.getShape(), m2.getShape());
            if (bShape != m1.getShape())
                m1 = m1.broadcast(bShape);
            if (bShape != m2.getShape())
                m2 = m2.broadcast(bShape);
        }

        template<typename T1, typename T2>
        static void broadcast(Matrix<T1> &m1, Matrix<T2> &m2, size_type dim) {
            if (dim <= 1) {
                broadcast(m1, m2);
                return;
            }
            if (dim > m1.getDims() && dim > m2.getDims())
                return;
            auto bShape = broadcastShape(m1.getShape(), m2.getShape(), dim);
            auto bShape1 = bShape;
            auto bShape2 = bShape;

            for (size_type i = 0; i < dim; ++i) {
                bShape1[i] = m1.getShape()[i];
                bShape2[i] = m2.getShape()[i];
            }
            m1 = m1.broadcast(bShape1);
            m2 = m2.broadcast(bShape2);
        }

        template<typename T1, typename T2>
        static void mul_broadcast(Matrix<T1> &m1, Matrix<T2> &m2) {
            if (m1.isEmpty() || m2.isEmpty())
                throw_broadcast_exception("Matrix multiplication cannot accept empty matrix");

            auto shape1 = newAxis(m1.getShape(), 2);
            auto shape2 = newAxis(m2.getShape(), 2);

            if (shape1[1] != shape2[0]) {
                if (shape1[1] == 1)
                    shape1[1] = shape2[0];
                else if (shape2[0] == 1)
                    shape2[0] = shape1[1];
                else
                    throw_broadcast_exception(MatrixBase::shapeToString(shape1), MatrixBase::shapeToString(shape2));
            }
            auto bShape = broadcastShape(shape1, shape2, 3);
            auto bShape1 = bShape;
            auto bShape2 = bShape;

            for (size_type i = 0; i < 2; ++i) {
                bShape1[i] = shape1[i];
                bShape2[i] = shape2[i];
            }
            m1 = m1.broadcast(bShape1);
            m2 = m2.broadcast(bShape2);
        }

        virtual auto operator()(size_type offset) -> dType & {
            if (offset < 0 || offset > size) {
                stringstream ss;
                ss << "read Matrix element out of Range: offset should be in size Range which( size = "
                   << size << " offset = " << offset;
                throw_out_of_range_error(ss.str());
            }
            return data[offset];
        }

        virtual auto operator()(size_type offset) const -> dType {
            if (offset < 0 || offset > size) {
                stringstream ss;
                ss << "read Matrix element out of Range: offset should be in size Range which( size = "
                   << size << " offset = " << offset;
                throw_out_of_range_error(ss.str());
            }
            return data[offset];
        }

        virtual /**
         * Taking the two-dimensional matrix as the smallest unit,
         * i and j are selected from the first and second dimensions respectively.
         * @param n The nth two-dimensional matrix
         * @param i i-th row (support negative index)
         * @param j j-th column (support negative index)
         * @return single element (reference)
         */
                auto
                operator()(size_type n, size_type i, size_type j) -> dType & {
            auto shape2d = newAxis(shape, 2);
            if (i < 0) i += shape2d[0];
            if (j < 0) j += shape2d[1];
            if (i < 0 || i >= shape2d[0])
                throw_out_of_range_error("read Matrix element out of Range: i(dim1) offset:" + to_string(i));
            if (j < 0 || j >= shape2d[1])
                throw_out_of_range_error("read Matrix element out of Range: j(dim2) offset:" + to_string(j));
            return data[n * getUnitSize() + i + j * shape2d[0]];
        }

        /**
         * Taking the two-dimensional matrix as the smallest unit,
         * i and j are selected from the first and second dimensions respectively.
         * @param n The nth two-dimensional matrix
         * @param i i-th row (support negative index)
         * @param j j-th column (support negative index)
         * @return single element (not reference)
         */
        virtual auto operator()(size_type n, size_type i, size_type j) const -> dType {
            auto shape2d = newAxis(shape, 2);
            if (i < 0) i += shape2d[0];
            if (j < 0) j += shape2d[1];
            if (i < 0 || i >= shape2d[0])
                throw_out_of_range_error("read Matrix element out of Range: i(dim1) offset:" + to_string(i));
            if (j < 0 || j >= shape2d[1])
                throw_out_of_range_error("read Matrix element out of Range: j(dim2) offset:" + to_string(j));
            return data[n * getUnitSize() + i + j * shape2d[0]];
        }

        /**
         * Taking the two-dimensional matrix as the smallest unit,
         * i and j are selected from the first and second dimensions respectively.
         * @param i i-th row (support negative index)
         * @param j j-th column (support negative index)
         * @return single element (reference)
         */
        virtual auto operator()(size_type i, size_type j) -> dType & {
            return (*this)(0, i, j);
        }

        /**
         * Taking the two-dimensional matrix as the smallest unit,
         * i and j are selected from the first and second dimensions respectively.
         * @param i i-th row (support negative index)
         * @param j j-th column (support negative index)
         * @return single element (not reference)
         */
        virtual auto operator()(size_type i, size_type j) const -> dType {
            return (*this)(0, i, j);
        }

        virtual auto operator()(shape_type pvec) -> dType & {
            if (pvec.size() != this->shape.size()) {
                stringstream ss;
                ss << "The number of dimensions of the read matrix elements does not match, The matrix dimensions:"
                   << shape.size() << " Input dimensions:" << pvec.size();
                throw_out_of_range_error(ss.str());
            }
            size_type dim = 0, pos = 0;
            for (auto &i: pvec) {
                if (i < 0) i += shape[dim];
                if (i < 0 || i >= shape[dim]) {
                    stringstream ss;
                    ss << "Reading matrix element, out of range In the" << dim + 1
                       << "-th dimension (shape) dimension length = " << shape[dim]
                       << " The selector dimension value i = " << i << " which is out of range";
                    throw_out_of_range_error(ss.str());
                }
                dim++;
            }
            for (auto iter = pvec.rbegin(); iter != pvec.rend(); ++iter)
                pos = *iter + pos * shape[--dim];
            return data[pos];
        }

        virtual auto operator()(shape_type pvec) const -> dType {
            if (pvec.size() != this->shape.size()) {
                stringstream ss;
                ss << "The number of dimensions of the read matrix elements does not match, The matrix dimensions:"
                   << shape.size() << " Input dimensions:" << pvec.size();
                throw_out_of_range_error(ss.str());
            }
            size_type dim = 0, pos = 0;
            for (auto &i: pvec) {
                if (i < 0) i += shape[dim];
                if (i < 0 || i >= shape[dim]) {
                    stringstream ss;
                    ss << "Reading matrix element, out of range In the" << dim + 1
                       << "-th dimension (shape) dimension length = " << shape[dim]
                       << " The selector dimension value i = " << i << " which is out of range";
                    throw_out_of_range_error(ss.str());
                }
                dim++;
            }
            for (auto iter = pvec.rbegin(); iter != pvec.rend(); ++iter)
                pos = *iter + pos * shape[--dim];
            return data[pos];
        }

        /**
         * @param svec Select some values from multiple dimensions.
         * If there is only one value and it is shape[dim] or -shape[dim], select all values.
         * @return selected items as a Matrix
         */
        auto operator()(vector<vector<size_type>> svec) const -> Matrix<dType> {
            if (svec.size() != this->shape.size()) {
                stringstream ss;
                ss << "The number of dimensions of the read matrix elements does not match, The matrix dimensions:"
                   << shape.size() << " Input dimensions:" << svec.size();
                throw_out_of_range_error(ss.str());
            }
            size_type dim = 0;
            shape_type r_shape;
            for (auto &v: svec) {
                if (v.size() == 1 && std::abs(v[0]) == shape[dim]) {
                    v = Util::VectorUtil::Range(shape[dim]);
                } else {
                    for (auto &i: v) {
                        if (i < -shape[dim] || i >= shape[dim]) {
                            stringstream ss;
                            ss << "Reading matrix element, out of range In the" << dim + 1
                               << "-th dimension (shape) dimension length = " << shape[dim]
                               << " The selector dimension value i = " << i << " which is out of range";
                            throw_out_of_range_error(ss.str());
                        }
                    }
                }
                r_shape.push_back((size_type) v.size());
                dim++;
            }
            Matrix<dType> r(r_shape, dType(0), this->getData());
            shape_type tracker(dim, 0);
            shape_type selector(dim, 0);
            auto iter_r = r.data, end_r = r.getDataEnd();
            size_type fresh = dim - 1;
            while (iter_r != end_r) {
                for (size_type i = 0; i <= fresh; ++i) {
                    selector[i] = svec[i][tracker[i]];
                }
                *iter_r = (*this)(selector);
                iter_r++;
                for (size_type i = 0; i < dim; ++i) {
                    if (++tracker[i] == r_shape[i])
                        tracker[i] = 0;
                    else {
                        fresh = i;
                        break;
                    }
                }
            }
            return r;
        }

        virtual auto operator+() const -> Matrix<dType> {
            return {*this};
        }

        virtual auto operator-() const -> Matrix<dType> {
            Matrix<dType> m(*this);
            // Negating a bool is identity; skip it (also avoids MSVC C4804 on Matrix<bool>).
            if constexpr (!std::is_same<dType, bool>::value)
                for (auto &i: m)
                    i = -i;
            return m;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator+=(const Matrix<Other> &right) -> Matrix & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            if (shape == right.getShape()) {
                Base::Add(data, right.getData(), size);
            } else {
                auto b_right = right;
                broadcast(*this, b_right);
                Base::Add(data, b_right.getData(), size);
            }
            return *this;
        }

        ///auto broadcast
        template<typename Other>
        auto operator+=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Add(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator-=(const Matrix<Other> &right) -> Matrix & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            if (shape == right.getShape()) {
                Base::Sub(data, right.getData(), size);
            } else {
                auto b_right = right;
                broadcast(*this, b_right);
                Base::Sub(data, b_right.getData(), size);
            }
            return *this;
        }

        ///auto broadcast
        template<typename Other>
        auto operator-=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Sub(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator*=(const Matrix<Other> &right) -> Matrix & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            if (shape == right.getShape()) {
                Base::Mul(data, right.getData(), size);
            } else {
                auto b_right = right;
                broadcast(*this, b_right);
                Base::Mul(data, b_right.getData(), size);
            }
            return *this;
        }

        ///auto broadcast
        template<typename Other>
        auto operator*=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Mul(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator/=(const Matrix<Other> &right) -> Matrix & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            if (shape == right.getShape()) {
                Base::Div(data, right.getData(), size);
            } else {
                auto b_right = right;
                broadcast(*this, b_right);
                Base::Div(data, b_right.getData(), size);
            }
            return *this;
        }

        ///auto broadcast
        template<typename Other>
        auto operator/=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Div(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto rem(const Matrix<Other> &right) -> Matrix & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            if (shape == right.getShape()) {
                Base::Rem(data, right.getData(), size);
            } else {
                auto b_right = right;
                broadcast(*this, b_right);
                Base::Rem(data, b_right.getData(), size);
            }
            return *this;
        }

        ///auto broadcast
        template<typename Other>
        auto rem(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Rem(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto mod(const Matrix<Other> &right) -> Matrix & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            if (shape == right.getShape()) {
                Base::Mod(data, right.getData(), size);
            } else {
                auto b_right = right;
                broadcast(*this, b_right);
                Base::Mod(data, b_right.getData(), size);
            }
            return *this;
        }

        ///auto broadcast
        template<typename Other>
        auto mod(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Mod(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator&=(const Matrix<Other> &right) -> Matrix & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            if (shape == right.getShape()) {
                Base::And(data, right.getData(), size);
            } else {
                auto b_right = right;
                broadcast(*this, b_right);
                Base::And(data, b_right.getData(), size);
            }
            return *this;
        }

        ///auto broadcast
        template<typename Other>
        auto operator&=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::And(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator|=(const Matrix<Other> &right) -> Matrix & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            if (shape == right.getShape()) {
                Base::Or(data, right.getData(), size);
            } else {
                auto b_right = right;
                broadcast(*this, b_right);
                Base::Or(data, b_right.getData(), size);
            }
            return *this;
        }

        ///auto broadcast
        template<typename Other>
        auto operator|=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Or(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator<(const Matrix<Other> &right) -> Matrix<bool> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<bool> r(b_left.shape, false);
                Base::Less(b_left.getData(), b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<bool> r(shape, false);
                Base::Less(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<typename Other>
        auto operator<(const Other &v) -> Matrix<bool> {
            Matrix<bool> r(shape, false);
            Base::Less(data, v, size, r.getData());
            return r;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator<=(const Matrix<Other> &right) -> Matrix<bool> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<bool> r(b_left.shape, 0.);
                Base::LessEqual(b_left.data, b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<bool> r(shape, false);
                Base::LessEqual(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<typename Other>
        auto operator<=(const Other &v) -> Matrix<bool> {
            Matrix<bool> r(shape, false);
            Base::LessEqual(data, v, size, r.getData());
            return r;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator>(const Matrix<Other> &right) -> Matrix<bool> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<bool> r(b_left.shape, 0.);
                Base::Great(b_left.data, b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<bool> r(shape, false);
                Base::Great(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<typename Other>
        auto operator>(const Other &v) -> Matrix<bool> {
            Matrix<bool> r(shape, false);
            Base::Great(data, v, size, r.getData());
            return r;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator>=(const Matrix<Other> &right) -> Matrix<bool> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<bool> r(b_left.shape, 0.);
                Base::GreatEqual(b_left.data, b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<bool> r(shape, false);
                Base::GreatEqual(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<typename Other>
        auto operator>=(const Other &v) -> Matrix<bool> {
            Matrix<bool> r(shape, false);
            Base::GreatEqual(data, v, size, r.getData());
            return r;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator==(const Matrix<Other> &right) -> Matrix<bool> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<bool> r(b_left.shape, 0.);
                Base::Equal(b_left.data, b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<bool> r(shape, false);
                Base::Equal(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<typename Other>
        auto operator==(const Other &v) -> Matrix<bool> {
            Matrix<bool> r(shape, false);
            Base::Equal(data, v, size, r.getData());
            return r;
        }

        ///BroadCast Needed!
        template<typename Other>
        auto operator!=(const Matrix<Other> &right) -> Matrix<bool> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<bool> r(b_left.shape, 0.);
                Base::NotEqual(b_left.data, b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<bool> r(shape, false);
                Base::NotEqual(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<typename Other>
        auto operator!=(const Other &v) -> Matrix<bool> {
            Matrix<bool> r(shape, false);
            Base::NotEqual(data, v, size, r.getData());
            return r;
        }

        /**
         * Tran the reciprocal of all numbers
         * Not considering precision
         */
        [[nodiscard]] auto num_inverse() const -> Matrix<dType> {
            Matrix<dType> m(*this);
            dType *iter = m.data, *end = m.data + m.size;
            while (iter != end) {
                *iter = (dType) 1. / *iter;
                iter++;
            }
            return m;
        }

        ///Auto predict shape
        [[nodiscard]] auto reshape(mshape_type new_shape) const -> Matrix<dType> {
            if (new_shape == shape)
                return {*this};
            shapePredict(size, new_shape);
            if (Util::VectorUtil::mul(new_shape) != size)
                throw_value_error("reshape new shape must have the same size of matrix shape:" +
                                  Util::VectorUtil::toString(shape) + " new shape=" +
                                  Util::VectorUtil::toString(new_shape));
            return {new_shape, data};
        }

        void expandShape(size_type dim) {
            this->setShape(newAxis(shape, dim));
        }

        void squeezeThis() {
            setShape(squeezeShape(shape));
        }

        [[nodiscard]] auto squeeze() const -> Matrix<dType> {
            if (shape.size() <= 2)
                return {*this};
            return {squeezeShape(shape), data};
        }

        void squeezeHighThis() {
            setShape(squeezeShapeHigh(shape));
        }

        [[nodiscard]] auto squeezeHigh() const -> Matrix<dType> {
            return {squeezeShapeHigh(shape), data};
        }

        /// Transpose: swaps the first two axes (applied to each 2-D page for N-D).
        [[nodiscard]] auto T() const -> Matrix<dType> {
            auto t_shape = shape;
            swap(t_shape[0], t_shape[1]);
            Matrix<dType> t(t_shape, 0.);
            size_type unit_size = t_shape[0] * t_shape[1];
            dType *iter = data, *iter_t = t.getData();
            for (size_type i = 0; i < size / unit_size; ++i) {
                Base::T(iter, iter_t, shape[0], shape[1]);
                iter += unit_size;
                iter_t += unit_size;
            }
            return t;
        }

        /**
         * This det uses the classic calculation formula, using det(Matrix) to quickly calculate using lu decomposition
         */
        [[nodiscard]] auto det() const -> dType {
            if (!is2D())
                throw_warning("det will only calculate first 2d-Matrix");
            if (!isSquare())
                throw_value_error("det only for square matrix");
            return Det(data);
        }

        ///FIXME usage of inv
        [[nodiscard]] auto inv() const -> Matrix<dType> {
            if (!isSquare()) {
                throw_value_error("Matrix inversion requires a square matrix");
                return {};
            }
            if (isEmpty())
                return {};
            if (isInt<dType>()) {
                throw_value_error(
                        "The matrix inversion self-function requires the original matrix to be at least"
                        " float32 float64 complex64 complex128. Please use inv(Matrix) for adaptation.");
                return {};
            }
            auto n = shape[0];
            Matrix<dType> m(*this);
            auto unit_size = n * n;
            auto unit_len = size / unit_size;
            auto unit_iter = m.getData(), unit_end = m.getDataEnd();
            while (unit_iter != unit_end) {
                inv(unit_iter);
                unit_iter += unit_size;
            }
            return m;
        }

        /**
         * is Square Matrix
         */
        [[nodiscard]] auto isSquare() const -> bool {
            return isEmpty() || (shape.size() > 1 && shape[0] == shape[1]);
        }

        /**
         * Only symmetric two-dimensional matrices are diag
         */
        [[nodiscard]] auto isDiag() const -> bool {
            auto m = squeezeHigh();
            // diag bind with 2d
            if (!m.is2D(true))
                return false;
            if (m.shape[0] != m.shape[1])
                return false;
            for (size_type i = 0; i < shape[0]; ++i) {
                for (size_type j = 0; j < shape[1]; ++j) {
                    if (i != j && m.getEle(i, j) != 0)
                        return false;
                }
            }
            return true;
        }

        [[nodiscard]] auto isEmpty() const -> bool {
            return getSize() == 0;
        }

        [[nodiscard]] auto isSingle() const -> bool {
            return getSize() == 1;
        }

        [[nodiscard]] auto isVec() const -> bool {
            return getDims() == 1;
        }

        [[nodiscard]] auto is2D() const -> bool {
            return getDims() == 2;
        }

        /**
         * @return matrix global max value
         */
        [[nodiscard]] auto Max() const -> dType {
            if (isEmpty()) return 0;
            dType r = *data;
            for (const auto &i: *this)
                if (i > r)
                    r = i;
            return r;
        }

        /**
         * @return matrix global min value
         */
        [[nodiscard]] auto Min() const -> dType {
            if (isEmpty()) return 0;
            dType r = *data;
            for (const auto &i: *this)
                if (i < r)
                    r = i;
            return r;
        }

        /**
         * @param up Default ascending order
         * @return Returns the global order of the copied matrix
         */
        [[nodiscard]] auto sort(bool up = true) const -> Matrix<dType> {
            if (isEmpty()) return {};
            Matrix<dType> r(*this);
            if (up)
                std::sort(r.begin(), r.end(),
                          [](const dType &left, const dType &right) { return left < right; });
            else
                std::sort(r.begin(), r.end(),
                          [](const dType &left, const dType &right) { return left > right; });
            return r;
        }

        [[nodiscard]] auto shapeToString(const string &segment = "x") const -> string {
            stringstream ss;
            ss << "(shape:"
               << Util::VectorUtil::toString(shape, segment)
               << ")";
            return ss.str();
        }

        /**
         * Print one-dimensional vectors as row vectors in sequence.
         * This is different from regular row and column printing.
         * This is printed in the form of data structure.
         */
        [[nodiscard]] auto toString(const string &matrixName = "Matrix", bool printMessage = true) const -> string {
            stringstream ss;
            if (printMessage) {
                ss << matrixName << "<" + getCalculableString<dType>() + ">  ";
                ss << shapeToString();
                ss << " =\n";
            }
            ss << arrayToString(0, 0) << ";\n";
            return ss.str();
        }

        /**
         * Print in classic row-column format, with each column being a one-dimensional column vector
         */
        [[nodiscard]] auto toRowColString(const string &matrixName = "Matrix", int dataSpace = 10) const -> string {
            stringstream ss;
            ss << "\n";

            auto unit_num = getUnitNum();
            shape_type unit_shape = getUnitShape();
            if (unit_num == 0) {
                ss << matrixName << " =\n\n";
                ss << setfill(' ') << setw(dataSpace) << "[]";
            } else if (unit_num == 1) {
                ss << matrixName << " =\n\n";
                for (size_type i = 0; i < unit_shape[0]; ++i) {
                    for (size_type j = 0; j < unit_shape[1]; ++j) {
                        auto p = getEle(i, j);
                        ss << dataToString(p, dataSpace) << ' ';
                    }
                    ss << "\n";
                }
            } else {
                shape_type shapeNum(getDims());
                for (size_type n = 0; n < getUnitNum(); ++n) {
                    //TODO calculate n -> shapeNum
                    for (size_type i = 2; i < getDims() - 1; ++i) {
                        if (shapeNum[i] < shape[i])
                            break;
                        shapeNum[i] = 0;
                        shapeNum[i + 1]++;
                    }
                    ss << matrixName << Util::VectorUtil::toString(shapeNum, ",") << " =\n\n";
                    shapeNum[2]++;
                    for (size_type i = 0; i < unit_shape[0]; ++i) {
                        for (size_type j = 0; j < unit_shape[1]; ++j) {
                            auto p = getEle(n, i, j);
                            ss << dataToString(p, dataSpace) << ' ';
                        }
                        ss << "\n";
                    }
                    ss << "\n";
                }
            }
            return ss.str();
        }

        /**
         * @author:jie
         * The magnitude of the vector
         */
        auto norm() -> float_max {
            float_max sum = 0.;
            for (const auto i: *this) {
                sum += i * i;
            }
            return sqrt(sum);
        }
    };
}
namespace matsar {
    /// Element-wise `+` with broadcasting (matrix+matrix, matrix+scalar, scalar+matrix).
    /// `-`, `*`, `/` follow the same pattern; `*` is element-wise, not a matrix product.
    template<typename lT, typename rT, typename TargetType = decltype(std::declval<lT>() + std::declval<rT>())>
    auto operator+(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp += right;
        return tmp;
    }

    template<typename lT, typename rT, typename TargetType = decltype(std::declval<lT>() + std::declval<rT>())>
    auto operator+(const Matrix<lT> &left, const rT &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp += right;
        return tmp;
    }

    template<typename lT, typename rT, typename TargetType = decltype(std::declval<lT>() + std::declval<rT>())>
    auto operator+(const lT &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(right);
        tmp += left;
        return tmp;
    }
}

namespace matsar {
    template<typename lT, typename rT, typename TargetType = decltype(std::declval<lT>() - std::declval<rT>())>
    auto operator-(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp -= right;
        return tmp;
    }

    template<typename lT, typename rT, typename TargetType = decltype(std::declval<lT>() - std::declval<rT>())>
    auto operator-(const Matrix<lT> &left, const rT &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp -= right;
        return tmp;
    }

    template<typename lT, typename rT, typename TargetType = decltype(std::declval<lT>() - std::declval<rT>())>
    auto operator-(const lT &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(-right);
        tmp += left;
        return tmp;
    }
}
namespace matsar {
    template<typename lT, typename rT, typename TargetType = decltype(std::declval<lT>() * std::declval<rT>())>
    auto mul(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp *= right;   // element-wise with broadcasting (consistent with operator*)
        return tmp;
    }

    template<typename lT, typename rT, typename TargetType = decltype((std::declval<lT>() + 0.) * (std::declval<rT>() + 0.))>
    auto operator*(const Matrix<lT> &left, const rT &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp *= TargetType{right + 0.};
        return tmp;
    }

    template<typename lT, typename rT, typename TargetType = decltype((std::declval<lT>() + 0.) * (std::declval<rT>() + 0.))>
    auto operator*(const lT &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(right);
        tmp *= TargetType{left + 0.};
        return tmp;
    }

    template<typename lT, typename rT, typename TargetType = decltype((std::declval<lT>() + 0.) * (std::declval<rT>() + 0.))>
    auto operator*(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp *= right;   // element-wise with broadcasting; matrix product is mmul()/operator%
        return tmp;
    }

#ifdef MATSAR_WITH_BLAS
    /**
     * Matrix product (linear algebra), BLAS-backed via OpenBLAS `*gemm`. Available
     * only when MATSAR_WITH_BLAS is defined. Inner dimensions must agree:
     * (m x k) * (k x n) -> (m x n). `operator%` is an alias for this.
     *
     * Note: `operator*` between two matrices is ELEMENT-wise (see operator+), not
     * this matrix product.
     */
    template<typename lT, typename rT, typename TargetType = decltype((std::declval<lT>() + 0.) * (std::declval<rT>() + 0.))>
    auto mmul(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        auto b_left = Matrix<TargetType>(left);
        auto b_right = Matrix<TargetType>(right);
        Matrix<TargetType>::mul_broadcast(b_left, b_right);
        auto r_shape = b_left.getShape();
        r_shape[1] = b_right.getShape()[1];
        Matrix<TargetType> r(r_shape, 0.);
        auto m = b_left.getShape()[0], n = b_right.getShape()[1], k = b_left.getShape()[1];
        MatrixBase::MMul(b_left.getData(), b_right.getData(), r.getData(), m, n, k, b_left.getUnitNum());
        return r;
    }

    /// Alias for mmul(): the BLAS-backed matrix product.
    template<typename lT, typename rT, typename TargetType = decltype((std::declval<lT>() + 0.) * (std::declval<rT>() + 0.))>
    auto operator%(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        return mmul<lT, rT, TargetType>(left, right);
    }
#endif // MATSAR_WITH_BLAS

    /**
     * Vector Dot Product
     * Vector horizontal input, one row represents a vector X or Y
     */
    template<typename dType, typename TargetType = decltype(std::declval<dType>() + std::declval<real_max>())>
    auto dot(Matrix<dType> A, Matrix<dType> B) -> Matrix<TargetType> {
        A = A.squeeze();
        B = B.squeeze();
        if (A.isEmpty() || B.isEmpty())
            return {};

        //0-D scalar
        if (A.isSingle() && B.isSingle())
            return {dType(A) * dType(B)};
        if (A.isSingle())
            return {dType(A) * B};
        if (B.isSingle())
            return {A * dType(B)};

        //vec inner product
        if ((A.isVec() || B.isVec()) && A.getShape()[0] != B.getShape()[0])
            throw_shape_exception("matrix A dim 0:" + to_string(A.getShape()[0]) +
                                  " not equals to matrix dim 0:" + to_string(B.getShape()[0]));

        // directly multiply and sum
        if (A.isVec() && B.isVec()) {
            TargetType sum = 0.;
            auto iter_A = A.begin();
            auto iter_B = B.begin();
            while (iter_A != A.end() && iter_B != B.end()) {
                sum += (*iter_A) * (*iter_B);
                iter_A++;
                iter_B++;
            }
            return {sum};
        }

        if (A.isVec() || B.isVec()) {
            auto &vec = A.isVec() ? A : B;
            auto &mm = A.isVec() ? B : A;

            auto r_shape = mm.getShape();
            r_shape[0] = 1;
            Matrix<TargetType> r(r_shape, 0.);
            auto iter_r = r.begin();
            auto iter_mm = mm.begin();
            while (iter_r != r.end()) {
                auto iter_vec = vec.begin();
                while (iter_vec != vec.end()) {
                    *iter_r += *iter_vec * (*iter_mm);
                    iter_vec++;
                    iter_mm++;
                }
            }
            return r;
        }

        //Two-dimensional matrix multiplication
        if (A.is2D() && B.is2D()) {
            return mmul(A, B);
        }

        //M-dimensional matrix and N-dimensional matrix
        if (A.getDims() < 2 || B.getDims() < 2) {
            A.debug();
            B.debug();
            throw_runtime_error_exception("dot failed");
        }

        if (A.getShape()[0] != B.getShape()[1])
            throw_shape_exception("matrix A dim 0:" + to_string(A.getShape()[0]) +
                                  " not equals to matrix B dim 1:" + to_string(B.getShape()[1]));
        else {
            auto t_B = B.T();
            auto r_shape = mshape_type();
            r_shape.insert(r_shape.end(), t_B.begin() + 1, t_B.end());
            r_shape.insert(r_shape.end(), A.begin() + 1, A.end());
            Matrix<TargetType> r(r_shape, 0.);

            auto iter_r = r.begin();
            auto A_vec_nums = A.getSize() / A.getShape()[0];
            auto tB_vec_nums = t_B.getSize() / t_B.getShape()[0];
            auto vec_size = A.getShape()[0];

            for (msize_type i = 0; i < A_vec_nums; ++i) {
                for (msize_type j = 0; j < tB_vec_nums; ++j) {
                    auto iter_A = A.begin() + i * vec_size;
                    auto iter_tB = t_B.begin() + j * vec_size;
                    for (msize_type k = 0; k < vec_size; ++k) {
                        *iter_r += *iter_A * *iter_tB;
                    }
                    iter_r++;
                }
            }
            return r;
        }
    }
}
namespace matsar {
    template<typename lT, typename rT, typename TargetType = decltype(std::declval<lT>() / std::declval<rT>() + std::declval<real_max>())>
    auto operator/(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp /= right;
        return tmp;
    }

    template<typename lT, typename rT, typename TargetType = decltype(std::declval<lT>() / std::declval<rT>() + std::declval<real_max>())>
    auto operator/(const Matrix<lT> &left, const rT &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp /= right;
        return tmp;
    }

    template<typename lT, typename rT, typename TargetType = decltype(std::declval<lT>() / std::declval<rT>() + std::declval<real_max>())>
    auto operator/(const lT &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(right.getShape(), static_cast<TargetType>(left));
        tmp /= right;
        return tmp;
    }
}
namespace matsar {
    template<typename T1, typename T2, typename TargetType = decltype(std::declval<T1>() & std::declval<T2>())>
    auto operator&(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp &= right;
        return tmp;
    }

    template<typename T1, typename T2, typename TargetType = decltype(std::declval<T1>() & std::declval<T2>())>
    auto operator&(const Matrix<T1> &left, const T2 &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp &= right;
        return tmp;
    }

    template<typename T1, typename T2, typename TargetType = decltype(std::declval<T1>() & std::declval<T2>())>
    auto operator&(const T1 &left, const Matrix<T2> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(right);
        tmp &= left;
        return tmp;
    }

    template<typename T1, typename T2, typename TargetType = decltype(std::declval<T1>() | std::declval<T2>())>
    auto operator|(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp |= right;
        return tmp;
    }

    template<typename T1, typename T2, typename TargetType = decltype(std::declval<T1>() | std::declval<T2>())>
    auto operator|(const Matrix<T1> &left, const T2 &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        tmp |= right;
        return tmp;
    }

    template<typename T1, typename T2, typename TargetType = decltype(std::declval<T1>() | std::declval<T2>())>
    auto operator|(const T1 &left, const Matrix<T2> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(right);
        tmp |= left;
        return tmp;
    }

    template<typename T1, typename T2, typename TargetType = decltype(std::declval<T1>() ^ std::declval<T2>())>
    auto bitxor(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<TargetType> {
        if (left.getShape() == right.getShape()) {
            Matrix<TargetType> tmp(left);
            MatrixBase::Xor(tmp.getData(), right.getData(), tmp.getSize());
            return tmp;
        } else {
            Matrix<TargetType> b_left(left);
            Matrix<TargetType> b_right(right);
            Matrix<TargetType>::broadcast(b_left, b_right);
            MatrixBase::Xor(b_left.getData(), b_right.getData(), b_left.getSize());
            return b_left;
        }
    }

    template<typename T1, typename T2, typename TargetType = decltype(std::declval<T1>() ^ std::declval<T2>())>
    auto bitxor(const Matrix<T1> &left, const T2 &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(left);
        MatrixBase::Xor(tmp.getData(), right, tmp.getSize());
        return tmp;
    }

    template<typename T1, typename T2, typename TargetType = decltype(std::declval<T1>() ^ std::declval<T2>())>
    auto bitxor(const T1 &left, const Matrix<T2> &right) -> Matrix<TargetType> {
        Matrix<TargetType> tmp(right);
        MatrixBase::Xor(tmp.getData(), left, tmp.getSize());
        return tmp;
    }
}

namespace matsar {
    template<typename lT, typename rT, typename TargetType = decltype(std::pow(std::declval<lT>(), std::declval<rT>()))>
    auto operator^(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<TargetType> {
        if (left.getShape() == right.getShape()) {
            Matrix<TargetType> r(left.getShape(), 0.);
            MatrixBase::Pow(left.getData(), right.getData(), r.getSize(), r.getData());
            return r;
        } else {
            auto b_left = left;
            auto b_right = right;
            Matrix<TargetType>::broadcast(b_left, b_right);
            Matrix<TargetType> r(b_left.getShape(), 0.);
            MatrixBase::Pow(b_left.getData(), b_right.getData(), r.getSize(), r.getData());
            return r;
        }
    }

    template<typename lT, typename rT, typename TargetType = decltype(std::pow(std::declval<lT>(), std::declval<rT>()))>
    auto operator^(const Matrix<lT> &left, const rT &right) -> Matrix<real_max> {
        Matrix<TargetType> r(left.getShape(), 0.);
        MatrixBase::Pow(left.getData(), right, r.getSize(), r.getData());
        return r;
    }

    template<typename lT, typename rT, typename TargetType = decltype(std::pow(std::declval<lT>(), std::declval<rT>()))>
    auto operator^(const lT &left, const Matrix<rT> &right) -> Matrix<real_max> {
        Matrix<TargetType> r(right.getShape(), 0.);
        MatrixBase::Pow(left, right.getData(), r.getSize(), r.getData());
        return r;
    }
}

namespace matsar {
    template<typename T>
    auto operator+(const Matrix<T> &left, const Matrix<bool> &right) -> Matrix<T> {
        Matrix<T> tmp_l(left), tmp_r(right);
        return tmp_l + tmp_r;
    }

    template<typename T>
    auto operator+(const Matrix<bool> &left, const Matrix<T> &right) -> Matrix<T> {
        Matrix<T> tmp_l(left), tmp_r(right);
        return tmp_l + tmp_r;
    }

    template<typename T = int_max>
    auto operator+(const Matrix<bool> &left, const Matrix<bool> &right) -> Matrix<T> {
        Matrix<T> tmp_l(left), tmp_r(right);
        return tmp_l + tmp_r;
    }
}

namespace matsar {
    template<typename T>
    auto operator-(const Matrix<T> &left, const Matrix<bool> &right) -> Matrix<T> {
        Matrix<T> tmp_l(left), tmp_r(right);
        return tmp_l - tmp_r;
    }

    template<typename T>
    auto operator-(const Matrix<bool> &left, const Matrix<T> &right) -> Matrix<T> {
        Matrix<T> tmp_l(left), tmp_r(right);
        return tmp_l - tmp_r;
    }

    template<typename T = int_max>
    auto operator-(const Matrix<bool> &left, const Matrix<bool> &right) -> Matrix<T> {
        Matrix<T> tmp_l(left), tmp_r(right);
        return tmp_l - tmp_r;
    }
}

namespace matsar {
    template<typename T>
    auto operator*(const Matrix<T> &left, const Matrix<bool> &right) -> Matrix<T> {
        Matrix<T> tmp_l(left), tmp_r(right);
        return tmp_l * tmp_r;
    }

    template<typename T>
    auto operator*(const Matrix<bool> &left, const Matrix<T> &right) -> Matrix<T> {
        Matrix<T> tmp_l(left), tmp_r(right);
        return tmp_l * tmp_r;
    }

    template<typename T = int_max>
    auto operator*(const Matrix<bool> &left, const Matrix<bool> &right) -> Matrix<T> {
        Matrix<T> tmp_l(left), tmp_r(right);
        return tmp_l * tmp_r;
    }
}

#endif //MATSAR_MATRIX_HPP