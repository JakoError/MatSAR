//
// Created by JakoError on 2022/7/2.
//
#ifndef MATSAR_MATRIX_HPP
#define MATSAR_MATRIX_HPP

#include <cstddef>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
//#include <xmemory>
#include <cstdarg>
#include <iomanip>

#include "util/Utils"
#include "NumberOperation.hpp"

#include <cblas.h>

#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#ifndef lapack_int
#define lapack_int int
#endif

#ifndef _LAPACKE_H_

#include <lapacke.h>

#endif

#define LIST_CONSTRUCTOR_MAKER(LIST_n) \
Matrix(LIST_n list) {\
    Util::VectorUtil::checkListFix(list);\
    this->size = Util::VectorUtil::getFixListSize(list); \
    this->shape = Util::VectorUtil::getDimSize(list);\
    allocate(this->size);\
    Util::VectorUtil::uninitListCopy(list, this->data);\
}

#define LIST_CONSTRUCTOR_MAKER_N \
LIST_CONSTRUCTOR_MAKER(LIST##2)\
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
VEC_CONSTRUCTOR_MAKER(VEC##2)\
VEC_CONSTRUCTOR_MAKER(VEC##3)\
VEC_CONSTRUCTOR_MAKER(VEC##4)\
VEC_CONSTRUCTOR_MAKER(VEC##5)\
VEC_CONSTRUCTOR_MAKER(VEC##6)\
VEC_CONSTRUCTOR_MAKER(VEC##7)\
VEC_CONSTRUCTOR_MAKER(VEC##8)

#define MATRIX_SIZE_TYPE int64
using msize_type = MATRIX_SIZE_TYPE;
using mshape_type = vector<msize_type>;

using namespace std;

using namespace MatSAR::Util::TypeUtil;
using MatSAR::Util::TypeUtil::getCalculableString;
using MatSAR::Util::TypeUtil::typeHigher;

using namespace MatSAR::Util::VectorUtil;
using MatSAR::Util::VectorUtil::mul;
using MatSAR::Util::VectorUtil::toString;
using MatSAR::Util::VectorUtil::to_string;

using namespace MatSAR::Util::Exception;

namespace MatSAR {
    template<Calculable dType>
    void swap(dType &left, dType &right) {
        auto tmp = left;
        left = right;
        right = tmp;
    }

    class MatrixBase {
    protected:
        using size_type = msize_type;

    public:
        static void checkShape(const mshape_type &shape) {
            for (const auto &i: shape) {
                if (i <= 0)
                    throw_value_error("维度不能为负或0 shape = " + toString(shape));
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

        static auto newAxis(const mshape_type &shape, int64 dim) -> mshape_type {
            checkShape(shape);
            if (shape.empty())
                return {};
            if (shape.size() > dim)
                return shape;
            mshape_type new_shape(shape);
            MatSAR::Util::VectorUtil::alignNum(new_shape, dim, (int64) 1);
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
         * 只去除shape高维度的1
         */
        static auto squeezeShapeHigh(const mshape_type &shape) -> mshape_type {
            checkShape(shape);
            mshape_type squeezed_shape;
            int64 idx = 0;
            for (int64 i = 0; i < shape.size(); ++i)
                if (shape[shape.size() - i - 1] != 1) {
                    idx = (int64) shape.size() - i - 1;
                    break;
                }
            for (int64 i = 0; i <= idx; ++i)
                squeezed_shape.push_back(shape[i]);
            if (squeezed_shape.empty())
                squeezed_shape.push_back(1);
            return squeezed_shape;
        }

        static auto broadcastShape(mshape_type shape1, mshape_type shape2, int64 dim = 1) -> mshape_type {
            auto dim_idx = dim - 1;
            alignNum(shape1, (int64) max(shape1.size(), shape2.size()), (int64) 1);
            alignNum(shape2, (int64) max(shape1.size(), shape2.size()), (int64) 1);
            mshape_type shape(shape1.size());
            for (int64 i = dim_idx; i < shape1.size(); ++i)
                if (shape1[i] == shape2[i] || shape2[i] == 1)
                    shape[i] = shape1[i];
                else if (shape1[i] == 1)
                    shape[i] = shape2[i];
                else
                    throw_broadcast_exception(toString(shape1), toString(shape2));
            return shape;
        }

        static void shapePredict(int target_size, mshape_type &shape) {
            auto iter = std::find(shape.begin(), shape.end(), -1);
            if (iter != shape.end()) {
                if (std::find(iter + 1, shape.end(), -1) != shape.end())//存在第二个-1
                    throw_value_error("矩阵维度只能声明一个未知维度(-1)");
                int64 exist_size = -MatSAR::Util::VectorUtil::mul(shape);
                if (target_size % exist_size != 0)//不能整除
                    throw_value_error("不能重构size=" + std::to_string(target_size) + "的数组到shape=" +
                                      MatSAR::Util::VectorUtil::toString(shape));
                *iter = target_size / exist_size;
            }
        };

        static auto shapeToString(const mshape_type &shape, const string &segment = "x") -> string {
            stringstream ss;
            ss << "(shape:";
            ss << Util::VectorUtil::toString(shape, segment);
            ss << ")";
            return ss.str();
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Add(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ += static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Add(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ += tmp;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Sub(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ -= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Sub(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ -= tmp;
        }

        ///NO CHECK!
        static void MMul(float32 *left, float32 *right, float32 *dst, size_type m, size_type n, size_type k,
                         size_type unit_times) {
            while (unit_times-- > 0) {
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, 1, left, m, right, k, 0, dst, m);
                left += m * k;
                right += k * n;
                dst += m * n;
            }
        }

        ///NO CHECK!
        static void MMul(float64 *left, float64 *right, float64 *dst, size_type m, size_type n, size_type k,
                         size_type unit_times) {
            while (unit_times-- > 0) {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, 1., left, m, right, k, 0, dst, m);

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
            while (unit_times-- > 0) {
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, &one, left, m, right, k, &zero, dst, m);
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
            while (unit_times-- > 0) {
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, &one, left, m, right, k, &zero, dst, m);
                left += m * k;
                right += k * n;
                dst += m * n;
            }
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Mul(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ *= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Mul(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ *= tmp;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Div(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ /= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Div(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ /= tmp;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Rem(leftT *left, rightT *right, size_type n) {
            while (n-- > 0) {
                *left = rem(*left, static_cast<leftT>(*right));
                ++left, ++right;
            }
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Rem(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0) {
                *left = rem(*left, tmp);
                ++left;
            }
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Mod(leftT *left, rightT *right, size_type n) {
            while (n-- > 0) {
                *left = mod(*left, static_cast<leftT>(*right));
                ++left, ++right;
            }
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Mod(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0) {
                *left = mod(*left, tmp);
                ++left;
            }
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void And(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ &= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void And(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ &= tmp;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Or(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ |= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Or(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ |= tmp;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Xor(leftT *left, rightT *right, size_type n) {
            while (n-- > 0)
                *left++ ^= static_cast<leftT>(*right++);
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT>
        static void Xor(leftT *left, const rightT &right, size_type n) {
            auto tmp = static_cast<leftT>(right);
            while (n-- > 0)
                *left++ ^= tmp;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void Less(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ < *right++;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void Less(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ < right;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void LessEqual(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ <= *right++;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void LessEqual(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ <= right;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void Great(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ > *right++;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void Great(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ > right;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void GreatEqual(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ >= *right++;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void GreatEqual(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ >= right;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void Equal(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ == *right++;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void Equal(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ == right;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void NotEqual(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ != *right++;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void NotEqual(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = *left++ != right;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void Pow(leftT *left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = std::pow(*left++, *right++);
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void Pow(leftT *left, const rightT &right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = std::pow(*left++, right);
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void Pow(const leftT &left, rightT *right, size_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = std::pow(left, *right++);
        }

    };

    template<Calculable dType>
    class Matrix : public MatrixBase {
    private:
        using alloc_trait = allocator_traits<allocator<dType>>;
    protected:
        using size_type = msize_type;
        using shape_type = mshape_type;
        using Base = MatrixBase;

        allocator<dType> alloc;
        dType *data = nullptr;
        dType *last = nullptr;
        size_type size = 0;
        mshape_type shape = {0};

        /**
         * 自动赋值data,last集中管理内存
         */
        auto allocate(int64 n, const void *hint = nullptr) -> dType * {
            try {
                if (hint)
                    data = alloc_trait::allocate(alloc, n, hint);
                else
                    data = alloc_trait::allocate(alloc, n);
                last = data + n;
                return data;
            } catch (const bad_alloc &e) {
                cerr << e.what() << ": " << "Unable to allocate" + to_string(n * sizeof(dType)) +
                                            "B memory for an array with shape " +
                                            MatSAR::Util::VectorUtil::toString(shape) + " and data type "
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
         * 自动赋值data,last集中管理内存
         */
        void deallocate() {
            try {
                if (data)
                    allocator_traits<allocator<dType>>::deallocate(alloc, data, size);
                data = last = nullptr;
            } catch (const exception &e) {
                cerr << e.what() << ": " << "Deallocate Failed!" << endl;
                debug();
            }
        }

        template<RealType RT>
        static auto dataToString(RT *&p, int dataSpace = 10) -> string {
            stringstream ss;
            ss.setf(ios::fixed);
            if (*p == 0.)
                ss.precision(0);
            else
                ss.precision(4);
            ss << setfill(' ') << setw(dataSpace) << *p;
            return ss.str();
        }

        template<ComplexType CT>
        static auto dataToString(CT *&p, int dataSpace = 10) -> string {
            stringstream ss;
            ss.setf(ios::fixed);
            ss.precision(4);
            ss << setfill(' ') << setw(dataSpace) << p->real();
            ss << " + ";
            ss << setfill(' ') << setw(dataSpace) << p->imag() << "i";
            return ss.str();
        }

        auto arrayToString(dType *&p, msize_type dim, int align = 0, int dataSpace = 10) const -> string {
            if (!p || dim < 0 || dim >= shape.size())
                return "{}";
            stringstream ss;
            string spaces(static_cast<std::basic_string<char>::size_type>((align + 1) * 2), ' ');
            if (dim == 0) {
                ss << spaces << "{";
                dType *end = p + shape[0];
                while (p != end) {
                    ss << dataToString(p);
                    if (++p == end)
                        ss << "}";
                    else
                        ss << ", ";
                }
                return ss.str();
            }
            ss << spaces << "{\n";
            for (int i = 0; i < shape[dim]; ++i) {
                ss << arrayToString(p, dim - 1, align + 1);
                if (i != shape[dim] - 1) {
                    ss << ",\n";
                    if (dim == 2)
                        ss << "\n";
                }
            }
            ss << "\n" << spaces << "}";
            return ss.str();
        }

        /**
         * single unit 2 dim matrix get dType& element
         * NO CHECK!
         */
        [[nodiscard]] auto getEle(dType *src, int64 i, int64 j) const -> dType & {
            return src[i + j * shape[0]];
        }

        /**
         * single unit 2 dim matrix get dType& element
         * NO CHECK!
         */
        [[nodiscard]] auto getEle(int64 n, int64 i, int64 j) const -> dType & {
            return (data + n * shape[0] * shape[1])[i + j * shape[0]];
        }

        /**
         * single unit 2 dim matrix get dType& element
         * NO CHECK!
         */
        [[nodiscard]] auto getEle(int64 i, int64 j) const -> dType & {
            return (data)[i + j * shape[0]];
        }

        /**
         * single unit 2 dim matrix
         * NO CHECK!
         */
        static void T(dType *src, dType *dst, int64 dim1, int64 dim2) {
            for (int64 i = 0; i < dim1; ++i)
                for (int64 j = 0; j < dim2; ++j)
                    *(dst++) = src[i + j * dim1];
        }

        /**
         * return a new Matrix for only 2d matrix
         * only works when row col all valid
         * FIXME 该方法的调动和参数可以商榷
         */
        auto remove_row_col(dType *src, int64 row, int64 col) const -> Matrix<dType> {
            if (row < 0 || col < 0 || row >= shape[0] || col >= shape[1])
                return {{shape[0], shape[1]}, src};
            if (shape[0] == 1 || shape[1] == 1)
                return {};
            vector<int64> removed_shape = {shape[0] - 1, shape[1] - 1};
            Matrix<dType> m(removed_shape, 0.);
            dType *iter_m = m.getData();
            for (int j = 0; j < shape[1]; ++j) {
                if (j == col) continue;
                for (int i = 0; i < shape[0]; ++i) {
                    if (i == col) continue;
                    *(iter_m++) = getEle(src, i, j);
                }
            }
            return m;
        }

        /**
         * single unit 2 dim matrix
         * 此函数可通过移动src来获取分页的二维度det
         */
        //FIXME 行列式运算太慢，是否可利用特征值求解
        auto det(dType *src) const -> dType {
            if (shape[0] == 1)
                return *src;
            int i = 0;
            dType ans = 0;
            for (int j = 0; j < shape[0]; ++j) {
                dType &num = getEle(src, i, j);
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
            int n = shape[0];
            unique_ptr<int64[]> is(new int64[n]);
            unique_ptr<int64[]> js(new int64[n]);
            int64 i, j, k;
            for (k = 0; k < n; ++k) {
                dType max = 0.;
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
                    throw_value_error("该矩阵不存在逆矩阵");
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

        template<Calculable T>
        static auto cat(size_type dim, vector<Matrix<T>> mv) -> Matrix<T> {
            if (dim < 0) throw_value_error("dim参数必须为正实数", true);
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
                    throw_value_error("cat各矩阵维度不匹配", true);
            }
            r_shape.insert(r_shape.begin() + dim_idx, dim_sum);
            Matrix<T> r(r_shape, 0.);
            if (dim == 1) {
                int64 dim_offset = 0;
                for (size_type mi = 0; mi < mv.size(); ++mi) {
                    vector<int64> selector(r_shape.size(), 0);
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
            ss << "Matrix debug message: " << " shape = " << MatSAR::Util::VectorUtil::toString(shape)
               << " size = " << size
               << " data = " << data
               << " matrix * = " << this;
            cerr << ss.str() << endl;
        }

        [[nodiscard]] auto getData() const -> dType * {
            return data;
        }

        [[nodiscard]] auto begin() const -> dType * {
            return data;
        }

        [[nodiscard]] auto getDataEnd() const -> dType * {
            return last;
        }

        [[nodiscard]] auto end() const -> dType * {
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
         * 矩阵最大维数的长度
         */
        [[nodiscard]] auto getLength() const -> size_type {
            return Util::VectorUtil::max(shape);
        }

        ///unit-二维矩阵单元
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

        ///unit-二维矩阵单元
        [[nodiscard]] auto getUnitNum() const -> size_type {
            if (size == 0)
                return 0;
            return size / getUnitSize();
        }

        ///unit-二维矩阵单元
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

        [[nodiscard]] auto real() const -> Matrix<float_max> {
            Matrix<float_max> r(shape, 0.);
            auto iter_c = data;
            for (auto *iter = r.getData(), *end = r.getDataEnd(); iter != end; ++iter, ++iter_c)
                *iter = std::real(*iter_c);
            return r;
        }

        [[nodiscard]] auto image() const -> Matrix<float_max> {
            Matrix<float_max> r(shape, 0.);
            auto iter_c = data;
            for (auto *iter = r.getData(), *end = r.getDataEnd(); iter != end; ++iter, ++iter_c)
                *iter = std::imag(*iter_c);
            return r;
        }

        [[nodiscard]] auto conj() const -> Matrix<complex_max> {
            Matrix<complex_max> r(shape, 0.);
            auto iter_c = data;
            for (auto *iter = r.getData(), *end = r.getDataEnd(); iter != end; ++iter, ++iter_c)
                *iter = std::conj(*iter_c);
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

        template<Calculable T>
        auto operator=(const Matrix<T> &m) -> Matrix<dType> & {
            this->operator=(Matrix<dType>(m));
            return *this;
        }

        auto operator=(const dType &m) -> Matrix<dType> & {
            this->operator=(Matrix<dType>({m}));
            return *this;
        }

        template<Calculable T>
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

        template<Calculable T>
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

        template<Calculable T>
        Matrix(const vector<Matrix<T>> &ms, int64 dim = 1) {
            if (typeHigher<T, dType>())
                throw_type_loss_exception<T, dType>(false);
            this->operator=(cat(dim, vector<Matrix<dType>>(ms)));
        };

        template<Calculable T>
        Matrix(const initializer_list<Matrix<T>> &ms, int64 dim = 1) {
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

//        template<Calculable T>
//        Matrix(const vector<T> &vec) {
//            this->shape = {vec.size()};
//            this->size = vec.size();
//            if (size != 0) {
//                allocate(size);
//                uninitialized_copy_n(vec.begin(), size, data);
//            }
//        }

        ///SINGLE VALUE GENERATOR
        explicit Matrix(const shape_type &m_shape, dType v,const void *hint = nullptr) {
            this->shape = Util::VectorUtil::abs(m_shape);
            this->size = Util::VectorUtil::mul(m_shape);
            if (size == 0) {
                this->shape = {0};
            } else {
                allocate(size,hint);
                uninitialized_fill_n(data, size, v);
            }
        };

        Matrix(const shape_type &m_shape, dType *v) {
            this->shape = Util::VectorUtil::abs(m_shape);
            this->size = Util::VectorUtil::mul(m_shape);
            if (size == 0) {
                this->shape = {0};
            } else {
                allocate(size,v);
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

        LIST_CONSTRUCTOR_MAKER_N

        Matrix(VEC1 vec) {
            allocate(vec.size());
            uninitialized_copy(vec.begin(), vec.end(), this->data);
            this->size = vec.size();
            this->shape = {this->size};
        }

        VEC_CONSTRUCTOR_MAKER_N

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
                            *(iter_r++) = *(iter_sqz + i);//重复读取sqz
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

        template<Calculable T1, Calculable T2>
        static void broadcast(Matrix<T1> &m1, Matrix<T2> &m2) {
            if (m1.getShape() == m2.getShape())
                return;
            auto bShape = broadcastShape(m1.getShape(), m2.getShape());
            if (bShape != m1.getShape())
                m1 = m1.broadcast(bShape);
            if (bShape != m2.getShape())
                m2 = m2.broadcast(bShape);
        }

        template<Calculable T1, Calculable T2>
        static void broadcast(Matrix<T1> &m1, Matrix<T2> &m2, int64 dim) {
            if (dim <= 1) {
                broadcast(m1, m2);
                return;
            }
            if (dim > m1.getDims() && dim > m2.getDims())
                return;
            auto bShape = broadcastShape(m1.getShape(), m2.getShape(), dim);
            auto bShape1 = bShape;
            auto bShape2 = bShape;

            for (int64 i = 0; i < dim; ++i) {
                bShape1[i] = m1.getShape()[i];
                bShape2[i] = m2.getShape()[i];
            }
            m1 = m1.broadcast(bShape1);
            m2 = m2.broadcast(bShape2);
        }

        template<Calculable T1, Calculable T2>
        static void mul_broadcast(Matrix<T1> &m1, Matrix<T2> &m2) {
            if (m1.isEmpty() || m2.isEmpty())
                throw_broadcast_exception("矩阵乘法不能接受空矩阵");

            auto shape1 = m1.getShape();
            auto shape2 = m2.getShape();

            newAxis(shape1,2);
            newAxis(shape2,2);

            if (shape1[1] != shape2[0]){
                if (shape1[1] == 1) shape1[1] = shape2[0];
                else if(shape2[0] == 1) shape2[0] = shape1[1];
                else
                    throw_broadcast_exception(MatrixBase::shapeToString(shape1),MatrixBase::shapeToString(shape2));
            }
            auto bShape = broadcastShape(shape1,shape2, 3);
            auto bShape1 = bShape;
            auto bShape2 = bShape;

            for (int64 i = 0; i < 2; ++i) {
                bShape1[i] = shape1[i];
                bShape2[i] = shape2[i];
            }
            m1 = m1.broadcast(bShape1);
            m2 = m2.broadcast(bShape2);
        }

//        /**
//         * 使用可变参数的重载会对参数要求比较高，不能出现比原有参数少或多的情况，\n
//         * 否则会出现越界，但这是一种内部可高效的访问方法，出现越界会异常退出\n
//         * 更加安全请使用Matrix({dim1,dim2,dim3,...,dimn})
//         * @param i
//         * @param ...
//         * @return
//         */
//        dType operator()(int64 i, ...) const {
//            va_list p;
//                    va_start(p, i);
//            int64 dim = 0, pos = 0;
//            array<int64, shape.size()> idx;
//            do {
//                if (i < 0)
//                    i += shape[dim];
//                ///如果该判断影响读取效率可去除，但进行判断很有必要///
//                if (i < 0 || i > this->shape[dim] - 1)
//                    throw_logical_error(
//                            "读取矩阵元素，超出矩阵范围 在" + to_string(dim + 1) + "维 (shape)维长=" + " 该维取值i=" + to_string(i) +
//                            "超出范围");
//                idx[dim] = i;
//            } while (++dim < shape.size());
//            for (int j = shape.size() - 1; j >= 0; --j)
//                pos = idx[j] + pos * shape[j];
//            return data[pos];
//        }
//
//        /**
//         * 使用可变参数的重载会对参数要求比较高，不能出现比原有参数少或多的情况，\n
//         * 否则会出现越界，但这是一种内部可高效的访问方法，出现越界会异常退出\n
//         * 更加安全请使用Matrix({dim1,dim2,dim3,...,dimn})
//         * @param i
//         * @param ...
//         * @return
//         */
//        dType &operator()(int64 i, ...) {
//            va_list p;
//                    va_start(p, i);
//            int64 dim = 0, pos = 0;
//            array<int64, shape.size()> idx;
//            do {
//                if (i < 0)
//                    i += shape[dim];
//                ///如果该判断影响读取效率可去除，但进行判断很有必要///
//                if (i < 0 || i > this->shape[dim] - 1)
//                    throw_logical_error(
//                            "读取矩阵元素，超出矩阵范围 在" + to_string(dim + 1) + "维 (shape)维长=" + " 该维取值i=" + to_string(i) +
//                            "超出范围");
//                idx[dim] = i;
//            } while (++dim < shape.size());
//            for (int j = shape.size() - 1; j >= 0; --j)
//                pos = idx[j] + pos * shape[j];
//            return data[pos];
//        }



        auto operator()(int64 offset) -> dType & {
            if (offset < 0 || offset > size) {
                stringstream ss;
                ss << "read Matrix element out of Range: offset should be in size Range which( size = "
                   << size << " offset = " << offset;
                throw_out_of_range_error(ss.str());
            }
            return data[offset];
        }

        auto operator()(int64 offset) const -> dType {
            if (offset < 0 || offset > size) {
                stringstream ss;
                ss << "read Matrix element out of Range: offset should be in size Range which( size = "
                   << size << " offset = " << offset;
                throw_out_of_range_error(ss.str());
            }
            return data[offset];
        }

        /**
         * 以二维矩阵作为最小单元，i，j分别从一维二维角度选择
         * @param n 第n个二维矩阵
         * @param i i行 支持负数倒数
         * @param j j列 支持负数倒数
         * @return 单个元素
         */
        auto operator()(int64 n, int64 i, int64 j) -> dType & {
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
         * 以二维矩阵作为最小单元，i，j分别从一维二维角度选择
         * @param n 第n个二维矩阵
         * @param i i行 支持负数倒数
         * @param j j列 支持负数倒数
         * @return 单个元素
         */
        auto operator()(int64 n, int64 i, int64 j) const -> dType {
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
        * 以二维矩阵作为最小单元，i，j分别从一维二维角度选择
        * @param i i行
        * @param j j列
        * @return 单个元素
        */
        auto operator()(int64 i, int64 j) -> dType & {
            return (*this)(0, i, j);
        }

        /**
        * 以二维矩阵作为最小单元，i，j分别从一维二维角度选择
        * @param i i行
        * @param j j列
        * @return 单个元素
        */
        auto operator()(int64 i, int64 j) const -> dType {
            return (*this)(0, i, j);
        }

//        dType operator()(initializer_list<int64> plist) const {
//            if (plist.size() != this->shape.size())
//                throw_out_of_range_error(
//                        "读取矩阵元素维度数不匹配 该矩阵维度:" + to_string(shape.size()) + " 传入维度:" + to_string(plist.size()), true);
//            int64 dim = 0, pos = 0;
//            for (const int64 &i: plist) {
//                if (i < -shape[dim] || i >= shape[dim]) {
//                    throw_out_of_range_error(
//                            "读取矩阵元素，超出矩阵范围 在" + to_string(i + 1) + "维 (shape)维长=" + " 该维取值i=" + to_string(i) + "超出范围",
//                            true);
//                }
//                dim++;
//            }
//            auto iter = rbegin(plist);
//            while (iter != rend(plist)) {
//                pos = *iter + pos * shape[--dim];
//                iter++;
//            }
//            return data[pos];
//        }
//
//        dType &operator()(initializer_list<int64> plist) {
//            if (plist.size() != this->shape.size())
//                throw_out_of_range_error(
//                        "读取矩阵元素维度数不匹配 该矩阵维度:" + to_string(shape.size()) + " 传入维度:" + to_string(plist.size()), true);
//            int64 dim = 0, pos = 0;
//            for (const int64 &i: plist) {
//                if (i < -shape[dim] || i >= shape[dim]) {
//                    throw_out_of_range_error(
//                            "读取矩阵元素，超出矩阵范围 在" + to_string(i + 1) + "维 (shape)维长=" + " 该维取值i=" + to_string(i) + "超出范围",
//                            true);
//                }
//                dim++;
//            }
//            auto iter = rbegin(plist);
//            while (iter != rend(plist)) {
//                pos = *iter + pos * shape[--dim];
//                iter++;
//            }
//            return data[pos];
//        }


        auto operator()(shape_type pvec) -> dType & {
            if (pvec.size() != this->shape.size()) {
                stringstream ss;
                ss << "读取矩阵元素维度数不匹配 该矩阵维度:" << shape.size() << " 传入维度:" << pvec.size();
                throw_out_of_range_error(ss.str());
            }
            int64 dim = 0, pos = 0;
            for (int64 &i: pvec) {
                if (i < 0) i += shape[dim];
                if (i < 0 || i >= shape[dim]) {
                    stringstream ss;
                    ss << "读取矩阵元素，超出矩阵范围 在" << dim + 1 << "维 (shape)维长 = " << shape[dim]
                       << " 该维取值i = " << i << "超出范围";
                    throw_out_of_range_error(ss.str());
                }
                dim++;
            }
            for (auto iter = rbegin(pvec); iter != rend(pvec); ++iter)
                pos = *iter + pos * shape[--dim];
            return data[pos];
        }

        auto operator()(shape_type pvec) const -> dType {
            if (pvec.size() != this->shape.size()) {
                stringstream ss;
                ss << "读取矩阵元素维度数不匹配 该矩阵维度:" << shape.size() << " 传入维度:" << pvec.size();
                throw_out_of_range_error(ss.str());
            }
            int64 dim = 0, pos = 0;
            for (int64 &i: pvec) {
                if (i < 0) i += shape[dim];
                if (i < 0 || i >= shape[dim]) {
                    stringstream ss;
                    ss << "读取矩阵元素，超出矩阵范围 在" << dim + 1 << "维 (shape)维长 = " << shape[dim]
                       << " 该维取值i = " << i << "超出范围";
                    throw_out_of_range_error(ss.str());
                }
                dim++;
            }
            for (auto iter = rbegin(pvec); iter != rend(pvec); ++iter)
                pos = *iter + pos * shape[--dim];
            return data[pos];
        }

        /**
         * @param svec 多个维度选取部分值，其中如果只有一个值并且为shape[dim] or -shape[dim]则选取所有
         * @return
         */
        auto operator()(vector<vector<size_type>> svec) const -> Matrix<dType> {
            if (svec.size() != this->shape.size()) {
                stringstream ss;
                ss << "读取矩阵元素维度数不匹配 该矩阵维度:" << shape.size() << " 传入维度:" << svec.size();
                throw_out_of_range_error(ss.str());
            }
            int64 dim = 0;
            shape_type r_shape;
            for (vector<size_type> &v: svec) {
                if (v.size() == 1 && std::abs(v[0]) == shape[dim])
                    v = Util::VectorUtil::Range(shape[dim]);
                else
                    for (int64 &i: v) {
                        if (i < -shape[dim] || i >= shape[dim]) {
                            stringstream ss;
                            ss << "读取矩阵，超出矩阵范围 在" << dim + 1 << "维 (shape)维长 = " << shape[dim]
                               << " 该维取值i = " << i << "超出范围";
                            throw_out_of_range_error(ss.str());
                        }
                    }
                r_shape.push_back((int64) v.size());
                dim++;
            }
            Matrix<dType> r(r_shape, 0.);
            vector<size_type> tracker(dim, 0);
            vector<size_type> selector(dim, 0);
            auto iter_r = r.data, end_r = r.getDataEnd();
            int64 fresh = dim - 1;
            while (iter_r != end_r) {
                for (int64 i = 0; i <= fresh; ++i)
                    selector[i] = svec[i][tracker[i]];
                *iter_r = (*this)(selector);
                iter_r++;
                for (int64 i = 0; i < dim; ++i) {
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

        ///BroadCast Needed!
        template<Calculable Other>
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
        template<Calculable Other>
        auto operator+=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Add(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<Calculable Other>
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
        template<Calculable Other>
        auto operator-=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Sub(data, v, size);
            return *this;
        }


//        ///Matirx Mul Special BroadCast!
//        template<Calculable Other>
//        auto operator*=(const Matrix<Other> &right) -> Matrix & {
//            if (Util::TypeUtil::typeHigher<Other, dType>())
//                throw_type_loss_exception<Other, dType>();
//            if (isEmpty() || right.isEmpty())
//                throw_value_error("矩阵乘法传入空矩阵");
//            auto left_shape = shape;
//            auto right_shape = right.getShape();
//            newAxis(left_shape, 2);
//            newAxis(right_shape, 2);
//            if (left_shape[1] != right_shape[0]) {
//                if (left_shape[1] == 1) left_shape[1] = right_shape[0];
//                else if (left_shape[0] == 1)right_shape[0] = left_shape[1];
//                else
//                    throw_broadcast_exception(MatrixBase::shapeToString(left_shape),
//                                              MatrixBase::shapeToString(right_shape));
//            }
//            broadcastShape(left_shape, right_shape, 3);
//            auto r_shape = left_shape;
//            r_shape[1] = right_shape[1];
//
//            auto m = left_shape[0], n = right_shape[1], k = left_shape[1];
//
//            Matrix r(r_shape, 0.);
//            if (shape == left_shape && right.getShape() == right_shape) {
//                Base::MMul(data, right.getData(), r.getData(), m, n, k, getUnitNum());
//            } else {
//                auto b_left = (*this).broadcast(left_shape);
//                auto b_right = right.broadcast(right_shape);
//                Base::MMul(b_left.getData(), b_right.getData(), r.getData(), m, n, k, b_left.getUnitNum());
//            }
//            return this->operator=(r);
//        }

        ///BroadCast Needed!
        template<Calculable Other>
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
        template<Calculable Other>
        auto operator*=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Mul(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<Calculable Other>
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
        template<Calculable Other>
        auto operator/=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Div(data, v, size);
            return *this;
        }

//        template<Calculable Other>
//        auto matmul(const Matrix<Other> &right) -> Matrix & {
//
//        }

//        ///矩阵乘法multiply 多维用pagemtimes
//        template<RealType RT>
//        auto operator*=(Matrix<RT> right) -> Matrix<dType> & {
//            if (Util::TypeUtil::typeHigher<RT, dType>())
//                throw_type_loss_exception<RT, dType>();
//            Matrix<dType> left(*this);
//            left.expandShape(2);
//            right.expandShape(2);
//            if (left.getDims() < 2 || right.getDims() < 2)
//                throw_value_error("Matrix multiply calculate at least or can expand to 2 dimension");
//            if (left.getShape()[1] != right.getShape()[0]) {
//                throw_value_error("Matrix multiply shape not match shape:" +
//                                  Util::VectorUtil::toString(left.getShape()) +
//                                  " shape:" + Util::VectorUtil::toString(right.getShape()));
//            }
//            broadcast(left, right, 3);
//            vector<int64> r_shape = left.getShape();
//            r_shape[1] = right.getShape()[1];
//            Matrix<dType> r(r_shape, 0.);
//
//            for (int64 n = 0; n < left.getUnitNum(); ++n)
//                for (int64 i = 0; i < r_shape[0]; ++i)
//                    for (int64 j = 0; j < r_shape[1]; ++j)
//                        for (int64 k = 0; k < left.getShape()[1]; ++k)
//                            r(n, i, j) += left(n, i, k) * right(n, k, j);
//            this->operator=(r);
//            return *this;
//        }
//
//        template<ComplexType CT>
//        Matrix<dType> &operator*=(Matrix<CT> right) {
//            if (Util::TypeUtil::typeHigher<CT, dType>()) {
//                throw_type_loss_exception<CT, dType>();
//            }
//            Matrix<dType> left(*this);
//            if (left.getShape().size() < 2 || right.getShape().size() < 2) {
//                throw_value_error("Matrix dot at least 2 dimension");
//            }
//            if (left.getShape()[1] != right.getShape()[0]) {
//                throw_value_error("Matrix dot shape not match shape:" +
//                                  Util::VectorUtil::toString(left.getShape()) +
//                                  " shape:" +
//                                  Util::VectorUtil::toString(right.getShape()));
//            }
//            if (!isComplex<dType>()) {
//                this->operator*=(right.real());
//            } else {
//                broadcast(left, right, 3);
//                vector<int64> r_shape = left.getShape();
//                r_shape[1] = right.getShape()[1];
//                Matrix<dType> r(r_shape, 0.);
//
//                for (int64 n = 0; n < left.getUnitNum(); ++n)
//                    for (int64 i = 0; i < r_shape[0]; ++i)
//                        for (int64 j = 0; j < r_shape[1]; ++j)
//                            for (int64 k = 0; k < left.getShape()[1]; ++k)
//                                r(n, i, j) += left(n, i, k) * right(n, k, j);
//                this->operator=(r);
//            }
//            return *this;
//        }


//        /**
//         * @redefine ^ not represent xor but power 重定义^为次幂运算
//         */
//        template<RealType T>
//        Matrix<dType> &operator^=(Matrix<T> right) {
//            broadcast(*this, right);
//            if (right.getShape() != shape) {
//                throw_value_error(
//                        "Matrix shape not match shape:" + Util::VectorUtil::toString(shape) + " shape:" +
//                        Util::VectorUtil::toString(right.getShape()));
//            }
//            if (Util::TypeUtil::typeHigher<T, dType>()) {
//                throw_type_loss_exception<T, dType>();
//            }
//            dType *iter = this->data, *end = this->data + this->size;
//            T *iter_right = right.getData();
//            while (iter != end)
//                *(iter++) = pow(*(iter++), *(iter_right++));
//            return *this;
//        }
//
//        ///BroadCast Needed!
//        /**
//         * @redefine ^ not represent xor but power 重定义^为次幂运算
//         */
//        template<ComplexType CT>
//        Matrix<dType> &operator^=(Matrix<CT> right) {
//            broadcast(*this, right);
//            if (right.getShape() != shape) {
//                throw_value_error(
//                        "Matrix shape not match shape:" + Util::VectorUtil::toString(shape) + " shape:" +
//                        Util::VectorUtil::toString(right.getShape()));
//            }
//            if (Util::TypeUtil::typeHigher<CT, dType>()) {
//                throw_type_loss_exception<CT, dType>();
//            }
//            if (!isComplex<dType>()) {
//                this->template operator^=(right.real());
//            } else {
//                dType *iter = this->data, *end = this->data + this->size;
//                CT *iter_right = right.getData();
//                while (iter != end)
//                    *(iter++) = pow(*(iter++), *(iter_right++));
//            }
//            return *this;
//        }
//
//        /**
//         * @redefine ^ not represent xor but power 重定义^为次幂运算
//         */
//        template<RealType T>
//        Matrix<dType> &operator^=(const T &v) {
//            if (Util::TypeUtil::typeHigher<T, dType>()) {
//                throw_type_loss_exception<T, dType>();
//            }
//            dType *iter = this->data, *end = this->data + this->size;
//            while (iter != end)
//                *(iter++) = pow(*(iter++), v);
//            return *this;
//        }
//
//        /**
//         * @redefine ^ not represent xor but power 重定义^为次幂运算
//         */
//        template<ComplexType CT>
//        Matrix<dType> &operator^=(const CT &v) {
//            if (Util::TypeUtil::typeHigher<CT, dType>()) {
//                throw_type_loss_exception<CT, dType>();
//            }
//            if (!isComplex<dType>()) {
//                this->template operator^=(v.real());
//            } else {
//                dType *iter = this->data, *end = this->data + this->size;
//                while (iter != end)
//                    *(iter++) = pow(*(iter++), v);
//            }
//            return *this;
//        }

        ///BroadCast Needed!
        template<Calculable Other>
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
        template<Calculable Other>
        auto rem(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Rem(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<Calculable Other>
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
        template<Calculable Other>
        auto mod(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Mod(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<Calculable Other>
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
        template<Calculable Other>
        auto operator&=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::And(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<Calculable Other>
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
        template<Calculable Other>
        auto operator|=(const Other &v) -> Matrix<dType> & {
            if (Util::TypeUtil::typeHigher<Other, dType>())
                throw_type_loss_exception<Other, dType>();
            Base::Or(data, v, size);
            return *this;
        }

        ///BroadCast Needed!
        template<Calculable Other>
        auto operator<(const Matrix<Other> &right) -> Matrix<int> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<int> r(b_left.shape, 0.);
                Base::Less(b_left.data, b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<int> r(shape, 0.);
                Base::Less(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<Calculable Other>
        auto operator<(const Other &v) -> Matrix<int> {
            Matrix<int> r(shape, 0.);
            Base::Less(data, v, size, r.getData());
            return r;
        }

        ///BroadCast Needed!
        template<Calculable Other>
        auto operator<=(const Matrix<Other> &right) -> Matrix<int> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<int> r(b_left.shape, 0.);
                Base::LessEqual(b_left.data, b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<int> r(shape, 0.);
                Base::LessEqual(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<Calculable Other>
        auto operator<=(const Other &v) -> Matrix<int> {
            Matrix<int> r(shape, 0.);
            Base::LessEqual(data, v, size, r.getData());
            return r;
        }

        ///BroadCast Needed!
        template<Calculable Other>
        auto operator>(const Matrix<Other> &right) -> Matrix<int> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<int> r(b_left.shape, 0.);
                Base::Great(b_left.data, b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<int> r(shape, 0.);
                Base::Great(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<Calculable Other>
        auto operator>(const Other &v) -> Matrix<int> {
            Matrix<int> r(shape, 0.);
            Base::Great(data, v, size, r.getData());
            return r;
        }

        ///BroadCast Needed!
        template<Calculable Other>
        auto operator>=(const Matrix<Other> &right) -> Matrix<int> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<int> r(b_left.shape, 0.);
                Base::GreatEqual(b_left.data, b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<int> r(shape, 0.);
                Base::GreatEqual(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<Calculable Other>
        auto operator>=(const Other &v) -> Matrix<int> {
            Matrix<int> r(shape, 0.);
            Base::GreatEqual(data, v, size, r.getData());
            return r;
        }

        ///BroadCast Needed!
        template<Calculable Other>
        auto operator==(const Matrix<Other> &right) -> Matrix<int> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<int> r(b_left.shape, 0.);
                Base::Equal(b_left.data, b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<int> r(shape, 0.);
                Base::Equal(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<Calculable Other>
        auto operator==(const Other &v) -> Matrix<int> {
            Matrix<int> r(shape, 0.);
            Base::Equal(data, v, size, r.getData());
            return r;
        }

        ///BroadCast Needed!
        template<Calculable Other>
        auto operator!=(const Matrix<Other> &right) -> Matrix<int> {
            if (shape != right.getShape()) {
                auto b_left = *this;
                auto b_right = right;
                broadcast(b_left, b_right);
                Matrix<int> r(b_left.shape, 0.);
                Base::NotEqual(b_left.data, b_right.getData(), b_left.size, r.getData());
                return r;
            } else {
                Matrix<int> r(shape, 0.);
                Base::NotEqual(data, right.getData(), size, r.getData());
                return r;
            }
        }

        ///auto broadcast
        template<Calculable Other>
        auto operator!=(const Other &v) -> Matrix<int> {
            Matrix<int> r(shape, 0.);
            Base::NotEqual(data, v, size, r.getData());
            return r;
        }

        /**
         * 全数求倒数
         * 不考虑精度
         */
        [[nodiscard]] auto num_inverse() const -> Matrix<dType> {
            Matrix<dType> m(*this);
            dType *iter = m.data, *end = m.data + m.size;
            while (iter != end) {
                *iter = 1. / *iter;
                iter++;
            }
            return m;
        }

        /**
         * 全数求负
         */
        [[nodiscard]] auto num_negative() const -> Matrix<dType> {
            Matrix<dType> m(*this);
            dType *iter = m.data, *end = m.data + m.size;
            while (iter != end) {
                *iter = -*iter;
                iter++;
            }
            return m;
        }

        [[nodiscard]] auto reshape(vector<int64> new_shape) const -> Matrix<dType> {
            if (new_shape == shape)
                return {*this};
            shapePredict(size, new_shape);
            if (Util::VectorUtil::mul(new_shape) != size)
                throw_value_error("reshape new shape must have the same size of matrix shape:" +
                                  Util::VectorUtil::toString(shape) + " new shape=" +
                                  Util::VectorUtil::toString(new_shape));
            return {new_shape, data};
        }

        void expandShape(int64 dim) {
            this->setShape(newAxis(shape, dim));
        }

        void squeezeThis() {
            if (shape.size() <= 2)
                return;
            setShape(squeezeShape(shape));
        }

        [[nodiscard]] auto squeeze() const -> Matrix<dType> {
            if (shape.size() <= 2)
                return {*this};
            return {squeezeShape(shape), data};
        }

        void squeezeHighThis() {
//            if (shape.size() <= 2)
//                return;
            setShape(squeezeShapeHigh(shape));
        }

        [[nodiscard]] auto squeezeHigh() const -> Matrix<dType> {
//            if (shape.size() <= 2)
//                return {*this};
            return {squeezeShapeHigh(shape), data};
        }

        [[nodiscard]] auto T() const -> Matrix<dType> {
            auto t_shape = shape;
            swap(t_shape[0], t_shape[1]);
            Matrix<dType> t(t_shape, 0.);
            size_type unit_size = t_shape[0] * t_shape[1];
            dType *iter = data, *iter_t = t.getData();
            for (int i = 0; i < size / unit_size; ++i) {
                T(iter, iter_t, shape[0], shape[1]);
                iter += unit_size;
                iter_t += unit_size;
            }
            return t;
        }

        /**
         * 此det使用的是经典计算公式，使用det(Matrix)使用lu分解快速计算
         */
        [[nodiscard]] auto det() const -> dType {
            if (!is2D())
                throw_warning("det will only calculate first 2d-Matrix");
            if (!isSquare())
                throw_value_error("det only for square matrix");
            return det(data);
        }

        ///FIXME inv使用方法
        [[nodiscard]] auto inv() const -> Matrix<dType> {
            if (shape.size() < 2 || shape[0] != shape[1]) {
                throw_value_error("矩阵取逆需要方阵");
                return {};
            }
            if (isInt<dType>()) {
                throw_value_error(
                        "矩阵取逆至少需要原矩阵为float32 float64 complex64 complex128,请使用inv(Matrix)进行自适应");
                return {};
            }
            int64 n = shape[0];
            Matrix<dType> m(*this);
            int64 unit_size = n * n;
            int64 unit_len = size / unit_size;
            dType *unit_iter = m.getData(), *unit_end = m.getDataEnd();
            while (unit_iter != unit_end) {
                inv(unit_iter);
                unit_iter += unit_size;
            }
            return m;
        }

        /**
         * 是否为方阵
         */
        [[nodiscard]] auto isSquare() const -> bool {
            return isEmpty() || (shape.size() > 1 && shape[0] == shape[1]);
        }

        /**
         * 只有对称的二维矩阵为diag
         */
        [[nodiscard]] auto isDiag() const -> bool {
            auto m = squeezeHigh();
            if (!m.is2D())
                return false;
            if (m.shape[0] != m.shape[1])
                return false;
            for (int i = 0; i < shape[0]; ++i) {
                for (int j = 0; j < shape[1]; ++j) {
                    if (i != j && m.getEle(i, j) != 0)
                        return false;
                }
            }
            return true;
        }

        [[nodiscard]] auto isEmpty() const -> bool {
            return data == nullptr;
        }

        [[nodiscard]] auto isSingle() const -> bool {
            return getSize() == 1;
        }

        [[nodiscard]] auto isVec() const -> bool {
            return getDims() == 1 || (squeezeHigh().getDims() == 1);
        }

        [[nodiscard]] auto is2D() const -> bool {
            return getDims() == 2 || (squeezeHigh().getDims() == 2);
        }

        /**
         * @return 返回矩阵全局最大值
         */
        [[nodiscard]] auto max() const -> dType {
            if (isEmpty()) return 0;
            dType r = *data;
            for (const auto &i: *this)
                if (i > r)
                    r = i;
            return r;
        }

        /**
         * @return 返回矩阵全局最小值
         */
        [[nodiscard]] auto min() const -> dType {
            if (isEmpty()) return 0;
            dType r = *data;
            for (const auto &i: *this)
                if (i < r)
                    r = i;
            return r;
        }

        /**
         * @param up 默认升序排列
         * @return 返回复制后矩阵的全局排序
         */
        [[nodiscard]] auto sort(bool up = true) const -> Matrix<dType> {
            if (isEmpty()) return {};
            Matrix<dType> r(*this);
            if (up)
                std::sort(r.begin(), r.end(),
                          [](const auto &left, const auto &right) { return left < right; });
            else
                std::sort(r.begin(), r.end(),
                          [](const auto &left, const auto &right) { return left > right; });
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
         * 以一维向量为行向量依次进行打印，与常规行列打印不同，此为数据结构形式打印
         */
        [[nodiscard]] auto toString(const string &matrixName = "Matrix", bool printMessage = true) const -> string {
            stringstream ss;
            if (printMessage) {
                ss << matrixName << "<" + getCalculableString<dType>() + ">  ";
                ss << shapeToString();
                ss << " =\n";
            }
            dType *p = this->data;
            ss << arrayToString(p, shape.size() - 1) << ";\n";
            return ss.str();
        }

        /**
         * 以经典行列形式打印，每列为一维列向量
         */
        [[nodiscard]] auto toRowColString(const string &matrixName = "Matrix") const -> string {
            int dataSpace = 10;//后置加入max长度
            stringstream ss;
            ss << "\n";

            int64 unit_num = getUnitNum();
            vector<int64> unit_shape = getUnitShape();
            if (unit_num == 0) {
                ss << matrixName << " =\n\n";
                ss << setfill(' ') << setw(dataSpace) << "[]";
            } else if (unit_num == 1) {
                ss << matrixName << " =\n\n";
                for (int64 i = 0; i < unit_shape[0]; ++i) {
                    for (int64 j = 0; j < unit_shape[1]; ++j) {
                        auto p = &getEle(i, j);
                        ss << dataToString(p);
                    }
                    ss << "\n";
                }
            } else {
                vector<int64> shapeNum(getDims());
                for (int64 n = 0; n < getUnitNum(); ++n) {
                    //TODO calculate n -> shapeNum
                    for (int i = 2; i < getDims() - 1; ++i) {
                        if (shapeNum[i] < shape[i])
                            break;
                        shapeNum[i] = 0;
                        shapeNum[i + 1]++;
                    }
                    ss << matrixName << Util::VectorUtil::toString(shapeNum, ",") << " =\n\n";
                    shapeNum[2]++;
                    for (int64 i = 0; i < unit_shape[0]; ++i) {
                        for (int64 j = 0; j < unit_shape[1]; ++j) {
                            auto p = &getEle(n, i, j);
                            ss << dataToString(p);
                        }
                        ss << "\n";
                    }
                    ss << "\n";
                }
            }
            return ss.str();
        }
    };

//    template<Calculable dType>
//    class SparseMatrix : public Matrix<dType>{
//    private:
//    protected:
//        using size_type = msize_type;
//
//    };
}
namespace MatSAR {
    template<Calculable dType>
    auto operator+(const Matrix<dType> &right) -> Matrix<dType> {
        return {right};
    }

    ///UInt相同类型运算
    template<UIntType lT, UIntType rT>
    auto operator+(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<uint_max> {
        Matrix<uint_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Int相同类型运算
    template<IntType lT, IntType rT>
    auto operator+(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Float相同类型运算
    template<FloatType lT, FloatType rT>
    auto operator+(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Complex相同类型运算
    template<ComplexType lT, ComplexType rT>
    auto operator+(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Uint SInt -> int_max
    template<UIntType lT, SIntType rT>
    auto operator+(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///(UInt,SInt)[IntType] Float -> float_max
    template<IntType lT, FloatType rT>
    auto operator+(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///(UInt,SInt,Float)[RealType] complex -> complex_max
    template<RealType lT, ComplexType rT>
    auto operator+(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        Matrix<real_max> real_left(left);
        Matrix<complex_max> tmp(real_left);
        tmp += right;
        return tmp;
    }

    ///SInt Uint -> int_max 交换律
    template<SIntType lT, UIntType rT>
    auto operator+(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Float (UInt,SInt)[IntType] -> float_max 交换律
    template<FloatType lT, IntType rT>
    auto operator+(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///complex (UInt,SInt,Float)[RealType]-> complex_max 交换律
    template<ComplexType lT, RealType rT>
    auto operator+(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///UInt相同类型运算
    template<UIntType lT, UIntType rT>
    auto operator+(const Matrix<lT> &left, const rT &right) -> Matrix<uint_max> {
        Matrix<uint_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Int相同类型运算
    template<IntType lT, IntType rT>
    auto operator+(const Matrix<lT> &left, const rT &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Float相同类型运算
    template<FloatType lT, FloatType rT>
    auto operator+(const Matrix<lT> &left, const rT &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Complex相同类型运算
    template<ComplexType lT, ComplexType rT>
    auto operator+(const Matrix<lT> &left, const rT &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Uint SInt -> int_max
    template<UIntType lT, SIntType rT>
    auto operator+(const Matrix<lT> &left, const rT &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///(UInt,SInt)[IntType] Float -> float_max
    template<IntType lT, FloatType rT>
    auto operator+(const Matrix<lT> &left, const rT &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///(UInt,SInt,Float)[RealType] complex -> complex_max
    template<RealType lT, ComplexType rT>
    auto operator+(const Matrix<lT> &left, const rT &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///SInt Uint -> int_max 交换律
    template<SIntType lT, UIntType rT>
    auto operator+(const Matrix<lT> &left, const rT &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Float (UInt,SInt)[IntType] -> float_max 交换律
    template<FloatType lT, IntType rT>
    auto operator+(const Matrix<lT> &left, const rT &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///complex (UInt,SInt,Float)[RealType]-> complex_max 交换律
    template<ComplexType lT, RealType rT>
    auto operator+(const Matrix<lT> &left, const rT &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///UInt相同类型运算 加法左右交换
    template<UIntType lT, UIntType rT>
    auto operator+(const lT &right, const Matrix<rT> &left) -> Matrix<uint_max> {
        Matrix<uint_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Int相同类型运算 加法左右交换
    template<IntType lT, IntType rT>
    auto operator+(const lT &right, const Matrix<rT> &left) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Float相同类型运算 加法左右交换
    template<FloatType lT, FloatType rT>
    auto operator+(const lT &right, const Matrix<rT> &left) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Complex相同类型运算 加法左右交换
    template<ComplexType lT, ComplexType rT>
    auto operator+(const lT &right, const Matrix<rT> &left) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Uint SInt -> int_max 加法左右交换
    template<UIntType lT, SIntType rT>
    auto operator+(const lT &right, const Matrix<rT> &left) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///(UInt,SInt)[IntType] Float -> float_max 加法左右交换
    template<IntType lT, FloatType rT>
    auto operator+(const lT &right, const Matrix<rT> &left) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///(UInt,SInt,Float)[RealType] complex -> complex_max 加法左右交换
    template<RealType lT, ComplexType rT>
    auto operator+(const lT &right, const Matrix<rT> &left) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///SInt Uint -> int_max 交换律 加法左右交换
    template<SIntType lT, UIntType rT>
    auto operator+(const lT &right, const Matrix<rT> &left) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///Float (UInt,SInt)[IntType] -> float_max 交换律 加法左右交换
    template<FloatType lT, IntType rT>
    auto operator+(const lT &right, const Matrix<rT> &left) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp += right;
        return tmp;
    }

    ///complex (UInt,SInt,Float)[RealType]-> complex_max 交换律 加法左右交换
    template<ComplexType lT, RealType rT>
    auto operator+(const lT &right, const Matrix<rT> &left) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp += right;
        return tmp;
    }
}

namespace MatSAR {
    template<Calculable T>
    auto operator-(const Matrix<T> &right) -> Matrix<T> {
        return right.num_negative();
    }

    ///UInt相同类型运算
    template<UIntType lT, UIntType rT>
    auto operator-(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<uint_max> {
        Matrix<uint_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///Int相同类型运算
    template<IntType lT, IntType rT>
    auto operator-(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///Float相同类型运算
    template<FloatType lT, FloatType rT>
    auto operator-(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///Complex相同类型运算
    template<ComplexType lT, ComplexType rT>
    auto operator-(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///Uint SInt -> int_max
    template<UIntType lT, SIntType rT>
    auto operator-(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///(UInt,SInt)[IntType] Float -> float_max
    template<IntType lT, FloatType rT>
    auto operator-(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///(UInt,SInt,Float)[RealType] complex -> complex_max
    template<RealType lT, ComplexType rT>
    auto operator-(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        Matrix<real_max> real_left(left);
        Matrix<complex_max> tmp(real_left);
        tmp -= right;
        return tmp;
    }

    ///SInt Uint -> int_max 交换律
    template<SIntType lT, UIntType rT>
    auto operator-(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///Float (UInt,SInt)[IntType] -> float_max 交换律
    template<FloatType lT, IntType rT>
    auto operator-(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///complex (UInt,SInt,Float)[RealType]-> complex_max 交换律
    template<ComplexType lT, RealType rT>
    auto operator-(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///UInt相同类型运算
    template<UIntType lT, UIntType rT>
    auto operator-(const Matrix<lT> &left, const rT &right) -> Matrix<uint_max> {
        Matrix<uint_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///Int相同类型运算
    template<IntType lT, IntType rT>
    auto operator-(const Matrix<lT> &left, const rT &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///Float相同类型运算
    template<FloatType lT, FloatType rT>
    auto operator-(const Matrix<lT> &left, const rT &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///Complex相同类型运算
    template<ComplexType lT, ComplexType rT>
    auto operator-(const Matrix<lT> &left, const rT &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///Uint SInt -> int_max
    template<UIntType lT, SIntType rT>
    auto operator-(const Matrix<lT> &left, const rT &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///(UInt,SInt)[IntType] Float -> float_max
    template<IntType lT, FloatType rT>
    auto operator-(const Matrix<lT> &left, const rT &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///(UInt,SInt,Float)[RealType] complex -> complex_max
    template<RealType lT, ComplexType rT>
    auto operator-(const Matrix<lT> &left, const rT &right) -> Matrix<complex_max> {
        Matrix<real_max> real_left(left);
        Matrix<complex_max> tmp(real_left);
        tmp -= right;
        return tmp;
    }

    ///SInt Uint -> int_max 交换律
    template<SIntType lT, UIntType rT>
    auto operator-(const Matrix<lT> &left, const rT &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///Float (UInt,SInt)[IntType] -> float_max 交换律
    template<FloatType lT, IntType rT>
    auto operator-(const Matrix<lT> &left, const rT &right) -> Matrix<float_max> {
        Matrix<float_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///complex (UInt,SInt,Float)[RealType]-> complex_max 交换律
    template<ComplexType lT, RealType rT>
    auto operator-(const Matrix<lT> &left, const rT &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp -= right;
        return tmp;
    }

    ///UInt相同类型运算 减法优化
    template<UIntType lT, UIntType rT>
    auto operator-(const lT &left, const Matrix<rT> &right) -> Matrix<uint_max> {
        return Matrix<uint_max>(right.num_negative()) += left;
    }

    ///Int相同类型运算 减法优化
    template<IntType lT, IntType rT>
    auto operator-(const lT &left, const Matrix<rT> &right) -> Matrix<int_max> {
        return Matrix<int_max>(right.num_negative()) += left;
    }

    ///Float相同类型运算
    template<FloatType lT, FloatType rT>
    auto operator-(const lT &left, const Matrix<rT> &right) -> Matrix<float_max> {
        return Matrix<float_max>(right.num_negative()) += left;
    }

    ///Complex相同类型运算
    template<ComplexType lT, ComplexType rT>
    auto operator-(const lT &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        return Matrix<complex_max>(right.num_negative()) += left;
    }

    ///Uint SInt -> int_max
    template<UIntType lT, SIntType rT>
    auto operator-(const lT &left, const Matrix<rT> &right) -> Matrix<int_max> {
        return Matrix<int_max>(right.num_negative()) += left;
    }

    ///(UInt,SInt)[IntType] Float -> float_max
    template<IntType lT, FloatType rT>
    auto operator-(const lT &left, const Matrix<rT> &right) -> Matrix<float_max> {
        return Matrix<float_max>(right.num_negative()) += left;
    }

    ///(UInt,SInt,Float)[RealType] complex -> complex_max
    template<RealType lT, ComplexType rT>
    auto operator-(const lT &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        return Matrix<complex_max>(right.num_negative()) += left;
    }

    ///SInt Uint -> int_max 交换律
    template<SIntType lT, UIntType rT>
    auto operator-(const lT &left, const Matrix<rT> &right) -> Matrix<int_max> {
        return Matrix<int_max>(right.num_negative()) += left;
    }

    ///Float (UInt,SInt)[IntType] -> float_max 交换律
    template<FloatType lT, IntType rT>
    auto operator-(const lT &left, const Matrix<rT> &right) -> Matrix<float_max> {
        return Matrix<float_max>(right.num_negative()) += left;
    }

    ///complex (UInt,SInt,Float)[RealType]-> complex_max 交换律
    template<ComplexType lT, RealType rT>
    auto operator-(const lT &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        return Matrix<complex_max>(right.num_negative()) += left;
    }
}
namespace MatSAR {
//    template<RealType T1, RealType T2>
//    auto mul(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<real_max> {
//        Matrix<real_max> r(left);
//        real_max *iter = r.getData(), *end = r.getDataEnd();
//        real_max *iter_right = right.getData();
//        while (iter != end)
//            *iter++ *= *iter_right++;
//        return r;
//    }
//
//    ///乘法交换律
//    template<RealType T1, ComplexType T2>
//    auto mul(const Matrix<T1> &right, const Matrix<T2> &left) -> Matrix<complex_max> {
//        Matrix<complex_max> r(left);
//        complex_max *iter = r.getData(), *end = r.getDataEnd();
//        real_max *iter_right = right.getData();
//        while (iter != end)
//            *iter++ *= *iter_right++;
//        return r;
//    }
//
//    template<ComplexType T1, RealType T2>
//    auto mul(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<complex_max> {
//        Matrix<complex_max> r(left);
//        complex_max *iter = r.getData(), *end = r.getDataEnd();
//        real_max *iter_right = right.getData();
//        while (iter != end)
//            *iter++ *= *iter_right++;
//        return r;
//    }
//
//    template<ComplexType T1, ComplexType T2>
//    auto mul(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<complex_max> {
//        Matrix<complex_max> r(left);
//        complex_max *iter = r.getData(), *end = r.getDataEnd();
//        complex_max *iter_right = right.getData();
//        while (iter != end)
//            *iter++ *= *iter_right++;
//        return r;
//    }

    template<RealType T1, RealType T2>
    auto mul(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<real_max> {
        Matrix<real_max> tmp(left);
        tmp *= right;
        return tmp;
    }

    template<RealType T1, ComplexType T2>
    auto mul(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp *= right;
        return tmp;
    }

    template<ComplexType T1, RealType T2>
    auto mul(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp *= right;
        return tmp;
    }

    template<ComplexType T1, ComplexType T2>
    auto mul(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp *= right;
        return tmp;
    }

    template<RealType T1, RealType T2>
    auto operator*(const Matrix<T1> &left, const T2 &right) -> Matrix<real_max> {
        Matrix<real_max> tmp(left);
        tmp *= right;
        return tmp;
    }

    template<RealType T1, ComplexType T2>
    auto operator*(const Matrix<T1> &left, const T2 &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp *= right;
        return tmp;
    }

    template<ComplexType T1, RealType T2>
    auto operator*(const Matrix<T1> &left, const T2 &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp *= right;
        return tmp;
    }

    template<ComplexType T1, ComplexType T2>
    auto operator*(const Matrix<T1> &left, const T2 &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp *= right;
        return tmp;
    }

    /**
     * law of commutation
     */
    template<RealType T1, RealType T2>
    auto operator*(const T1 &left, const Matrix<T2> &right) -> Matrix<real_max> {
        Matrix<real_max> tmp(right);
        tmp *= left;
        return tmp;
    }

    /**
     * law of commutation
     */
    template<RealType T1, ComplexType T2>
    auto operator*(const T1 &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(right);
        tmp *= left;
        return tmp;
    }

    /**
     * law of commutation
     */
    template<ComplexType T1, RealType T2>
    auto operator*(const T1 &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(right);
        tmp *= left;
        return tmp;
    }

    /**
     * law of commutation
     */
    template<ComplexType T1, ComplexType T2>
    auto operator*(const T1 &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(right);
        tmp *= left;
        return tmp;
    }

    template<RealType T1, RealType T2>
    auto operator*(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<real_max> {
        if (left.isEmpty() || right.isEmpty())
            throw_value_error("矩阵乘法传入空矩阵");
        auto b_left = Matrix<real_max>(left);
        auto b_right = Matrix<real_max>(right);
        Matrix<real_max>::mul_broadcast(b_left,b_right);
        auto r_shape = b_left.getShape();
        r_shape[1] = b_right.getShape()[1];
        Matrix<real_max> r(r_shape,0.);
        auto m = b_left.getShape()[0],n = b_right.getShape()[1],k = b_left.getShape()[1];
        MatrixBase::MMul(b_left.getData(),b_right.getData(),r.getData(),m,n,k,b_left.getUnitNum());
        return r;
    }

    template<RealType T1, ComplexType T2>
    auto operator*(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        if (left.isEmpty() || right.isEmpty())
            throw_value_error("矩阵乘法传入空矩阵");
        auto b_left = Matrix<complex_max>(left);
        auto b_right = Matrix<complex_max>(right);
        Matrix<complex_max>::mul_broadcast(b_left,b_right);
        auto r_shape = b_left.getShape();
        r_shape[1] = b_right.getShape()[1];
        Matrix<complex_max> r(r_shape,0.);
        auto m = b_left.getShape()[0],n = b_right.getShape()[1],k = b_left.getShape()[1];
        MatrixBase::MMul(b_left.getData(),b_right.getData(),r.getData(),m,n,k,b_left.getUnitNum());
        return r;
    }

    template<ComplexType T1, RealType T2>
    auto operator*(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        if (left.isEmpty() || right.isEmpty())
            throw_value_error("矩阵乘法传入空矩阵");
        auto b_left = Matrix<complex_max>(left);
        auto b_right = Matrix<complex_max>(right);
        Matrix<complex_max>::mul_broadcast(b_left,b_right);
        auto r_shape = b_left.getShape();
        r_shape[1] = b_right.getShape()[1];
        Matrix<complex_max> r(r_shape,0.);
        auto m = b_left.getShape()[0],n = b_right.getShape()[1],k = b_left.getShape()[1];
        MatrixBase::MMul(b_left.getData(),b_right.getData(),r.getData(),m,n,k,b_left.getUnitNum());
        return r;
    }

    template<ComplexType T1, ComplexType T2>
    auto operator*(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        if (left.isEmpty() || right.isEmpty())
            throw_value_error("矩阵乘法传入空矩阵");
        auto b_left = Matrix<complex_max>(left);
        auto b_right = Matrix<complex_max>(right);
        Matrix<complex_max>::mul_broadcast(b_left,b_right);
        auto r_shape = b_left.getShape();
        r_shape[1] = b_right.getShape()[1];
        Matrix<complex_max> r(r_shape,0.);
        auto m = b_left.getShape()[0],n = b_right.getShape()[1],k = b_left.getShape()[1];
        MatrixBase::MMul(b_left.getData(),b_right.getData(),r.getData(),m,n,k,b_left.getUnitNum());
        return r;
    }
}
namespace MatSAR {
    template<RealType T1, RealType T2>
    auto operator/(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<real_max> {
        Matrix<real_max> tmp(left);
        tmp /= right;
        return tmp;
    }

    template<RealType T1, ComplexType T2>
    auto operator/(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp /= right;
        return tmp;
    }

    template<ComplexType T1, RealType T2>
    auto operator/(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp /= right;
        return tmp;
    }

    template<ComplexType T1, ComplexType T2>
    auto operator/(const Matrix<T1> &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp /= right;
        return tmp;
    }

    template<RealType T1, RealType T2>
    auto operator/(const Matrix<T1> &left, const T2 &right) -> Matrix<real_max> {
        Matrix<real_max> tmp(left);
        tmp /= right;
        return tmp;
    }

    template<RealType T1, ComplexType T2>
    auto operator/(const Matrix<T1> &left, const T2 &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp /= right;
        return tmp;
    }

    template<ComplexType T1, RealType T2>
    auto operator/(const Matrix<T1> &left, const T2 &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp /= right;
        return tmp;
    }

    template<ComplexType T1, ComplexType T2>
    auto operator/(const Matrix<T1> &left, const T2 &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        tmp /= right;
        return tmp;
    }


    /**
     * be divided optimize
     */
    template<RealType T1, RealType T2>
    auto operator/(const T1 &left, const Matrix<T2> &right) -> Matrix<real_max> {
        Matrix<real_max> tmp(right);
        real_max left_ = left;
        for (auto &i: tmp)
            i = left_ / i;
        return tmp;
    }

    /**
     * be divided optimize
     */
    template<RealType T1, ComplexType T2>
    auto operator/(const T1 &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(right);
        complex_max left_ = (real_max) left;
        for (auto &i: tmp)
            i = left_ / i;
        return tmp;
    }

    /**
     * be divided optimize
     */
    template<ComplexType T1, RealType T2>
    auto operator/(const T1 &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(right);
        complex_max left_ = left;
        for (auto &i: tmp)
            i = left_ / i;
        return tmp;
    }

    /**
     * be divided optimize
     */
    template<ComplexType T1, ComplexType T2>
    auto operator/(const T1 &left, const Matrix<T2> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(right);
        complex_max left_ = left;
        for (auto &i: tmp)
            i = left_ / i;
        return tmp;
    }
}
namespace MatSAR {
    template<SIntType SIT1, SIntType SIT2>
    auto operator&(const Matrix<SIT1> &left, const Matrix<SIT2> &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp &= right;
        return tmp;
    }

    template<SIntType SIT1, SIntType SIT2>
    auto operator&(const Matrix<SIT1> &left, const SIT2 &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp &= right;
        return tmp;
    }

    template<SIntType SIT1, SIntType SIT2>
    auto operator&(const SIT1 &left, const Matrix<SIT2> &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(right);
        tmp &= left;
        return tmp;
    }

    template<UIntType UIT1, UIntType UIT2>
    auto operator&(const Matrix<UIT1> &left, const Matrix<UIT2> &right) -> Matrix<uint_max> {
        Matrix<int_max> tmp(left);
        tmp &= right;
        return tmp;
    }

    template<UIntType UIT1, UIntType UIT2>
    auto operator&(const Matrix<UIT1> &left, const UIT2 &right) -> Matrix<uint_max> {
        Matrix<int_max> tmp(left);
        tmp &= right;
        return tmp;
    }

    template<UIntType UIT1, UIntType UIT2>
    auto operator&(const UIT1 &left, const Matrix<UIT2> &right) -> Matrix<uint_max> {
        Matrix<int_max> tmp(right);
        tmp &= left;
        return tmp;
    }

    template<SIntType SIT1, SIntType SIT2>
    auto operator|(const Matrix<SIT1> &left, const Matrix<SIT2> &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp |= right;
        return tmp;
    }

    template<SIntType SIT1, SIntType SIT2>
    auto operator|(const Matrix<SIT1> &left, const SIT2 &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        tmp |= right;
        return tmp;
    }

    template<SIntType SIT1, SIntType SIT2>
    auto operator|(const SIT1 &left, const Matrix<SIT2> &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(right);
        tmp |= left;
        return tmp;
    }

    template<UIntType UIT1, UIntType UIT2>
    auto operator|(const Matrix<UIT1> &left, const Matrix<UIT2> &right) -> Matrix<uint_max> {
        Matrix<int_max> tmp(left);
        tmp |= right;
        return tmp;
    }

    template<UIntType UIT1, UIntType UIT2>
    auto operator|(const Matrix<UIT1> &left, const UIT2 &right) -> Matrix<uint_max> {
        Matrix<int_max> tmp(left);
        tmp |= right;
        return tmp;
    }

    template<UIntType UIT1, UIntType UIT2>
    auto operator|(const UIT1 &left, const Matrix<UIT2> &right) -> Matrix<uint_max> {
        Matrix<int_max> tmp(right);
        tmp |= left;
        return tmp;
    }

    template<SIntType SIT1, SIntType SIT2>
    auto bitxor(const Matrix<SIT1> &left, const Matrix<SIT2> &right) -> Matrix<int_max> {
        if (left.getShape() == right.getShape()) {
            Matrix<int_max> tmp(left);
            MatrixBase::Xor(tmp.getData(), right.getData(), tmp.getSize());
            return tmp;
        } else {
            Matrix<int_max> tmp(left);
            Matrix<int_max> b_right(right);
            Matrix<int_max>::broadcast(tmp, b_right);
            MatrixBase::Xor(tmp.getData(), b_right.getData(), tmp.getSize());
            return tmp;
        }
    }

    template<SIntType SIT1, SIntType SIT2>
    auto bitxor(Matrix<SIT1> left, const SIT2 &right) -> Matrix<int_max> {
        Matrix<int_max> tmp(left);
        MatrixBase::Xor(tmp.getData(), right, tmp.getSize());
        return tmp;
    }

    template<SIntType SIT1, SIntType SIT2>
    auto bitxor(const SIT1 &left, Matrix<SIT2> right) -> Matrix<int_max> {
        Matrix<int_max> tmp(right);
        MatrixBase::Xor(tmp.getData(), left, tmp.getSize());
        return tmp;
    }

    template<UIntType UIT1, UIntType UIT2>
    auto bitxor(const Matrix<UIT1> &left, const Matrix<UIT2> &right) -> Matrix<uint_max> {
        if (left.getShape() == right.getShape()) {
            Matrix<int_max> tmp(left);
            MatrixBase::Xor(tmp.getData(), right.getData(), tmp.getSize());
            return tmp;
        } else {
            Matrix<int_max> tmp(left);
            Matrix<int_max> b_right(right);
            Matrix<int_max>::broadcast(tmp, b_right);
            MatrixBase::Xor(tmp.getData(), b_right.getData(), tmp.getSize());
            return tmp;
        }
    }

    template<UIntType UIT1, UIntType UIT2>
    auto bitxor(Matrix<UIT1> left, const UIT2 &right) -> Matrix<uint_max> {
        Matrix<int_max> tmp(left);
        MatrixBase::Xor(tmp.getData(), right, tmp.getSize());
        return tmp;
    }

    template<UIntType UIT1, UIntType UIT2>
    auto bitxor(const UIT1 &left, Matrix<UIT2> right) -> Matrix<uint_max> {
        Matrix<int_max> tmp(right);
        MatrixBase::Xor(tmp.getData(), left, tmp.getSize());
        return tmp;
    }
}

namespace MatSAR {
    template<RealType lT, IntType rT>
    auto operator^(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<real_max> {
        if (left.getShape() == right.getShape()) {
            Matrix<real_max> tmp(left);
            MatrixBase::Pow(tmp.getData(), right.getData(), tmp.getSize(), tmp.getData());
            return tmp;
        } else {
            Matrix<real_max> tmp(left);
            auto b_right = right;
            Matrix<real_max>::broadcast(tmp, b_right);
            MatrixBase::Pow(tmp.getData(), b_right.getData(), tmp.getSize(), tmp.getData());
            return tmp;
        }
    }

    template<RealType lT, IntType rT>
    auto operator^(const Matrix<lT> &left, const rT &right) -> Matrix<real_max> {
        Matrix<real_max> tmp(left);
        MatrixBase::Pow(tmp.getData(), right, tmp.getSize(), tmp.getData());
        return tmp;
    }

    template<RealType lT, IntType rT>
    auto operator^(const lT &left, const Matrix<rT> &right) -> Matrix<real_max> {
        Matrix<real_max> tmp(right);
        MatrixBase::Pow(left, tmp.getData(), tmp.getSize(), tmp.getData());
        return tmp;
    }

    template<RealType lT, FloatType rT>
    auto operator^(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        if (left.getShape() == right.getShape()) {
            Matrix<complex_max> tmp(left);
            MatrixBase::Pow(tmp.getData(), right.getData(), tmp.getSize(), tmp.getData());
            return tmp;
        } else {
            Matrix<complex_max> tmp(left);
            auto b_right = right;
            Matrix<complex_max>::broadcast(tmp, b_right);
            MatrixBase::Pow(tmp.getData(), b_right.getData(), tmp.getSize(), tmp.getData());
            return tmp;
        }
    }

    template<RealType lT, FloatType rT>
    auto operator^(const Matrix<lT> &left, const rT &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        MatrixBase::Pow(tmp.getData(), right, tmp.getSize(), tmp.getData());
        return tmp;
    }

    template<RealType lT, FloatType rT>
    auto operator^(const lT &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(right);
        MatrixBase::Pow(left, tmp.getData(), tmp.getSize(), tmp.getData());
        return tmp;
    }

    template<ComplexType lT, Calculable rT>
    auto operator^(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        if (left.getShape() == right.getShape()) {
            Matrix<complex_max> tmp(left);
            MatrixBase::Pow(tmp.getData(), right.getData(), tmp.getSize(), tmp.getData());
            return tmp;
        } else {
            Matrix<complex_max> tmp(left);
            auto b_right = right;
            Matrix<complex_max>::broadcast(tmp, b_right);
            MatrixBase::Pow(tmp.getData(), b_right.getData(), tmp.getSize(), tmp.getData());
            return tmp;
        }
    }

    template<ComplexType lT, Calculable rT>
    auto operator^(const Matrix<lT> &left, const rT &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        MatrixBase::Pow(tmp.getData(), right, tmp.getSize(), tmp.getData());
        return tmp;
    }

    template<ComplexType lT, Calculable rT>
    auto operator^(const lT &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(right);
        MatrixBase::Pow(left, tmp.getData(), tmp.getSize(), tmp.getData());
        return tmp;
    }

    template<Calculable lT, ComplexType rT>
    auto operator^(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        if (left.getShape() == right.getShape()) {
            Matrix<complex_max> tmp(left);
            MatrixBase::Pow(tmp.getData(), right.getData(), tmp.getSize(), tmp.getData());
            return tmp;
        } else {
            Matrix<complex_max> tmp(left);
            auto b_right = right;
            Matrix<complex_max>::broadcast(tmp, b_right);
            MatrixBase::Pow(tmp.getData(), b_right.getData(), tmp.getSize(), tmp.getData());
            return tmp;
        }
    }

    template<Calculable lT, ComplexType rT>
    auto operator^(const Matrix<lT> &left, const rT &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(left);
        MatrixBase::Pow(tmp.getData(), right, tmp.getSize(), tmp.getData());
        return tmp;
    }

    template<Calculable lT, ComplexType rT>
    auto operator^(const lT &left, const Matrix<rT> &right) -> Matrix<complex_max> {
        Matrix<complex_max> tmp(right);
        MatrixBase::Pow(left, tmp.getData(), tmp.getSize(), tmp.getData());
        return tmp;
    }

}
#endif //MATSAR_MATRIX_HPP