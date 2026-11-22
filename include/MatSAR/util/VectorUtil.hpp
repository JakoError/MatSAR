// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2022/7/2.
//

#ifndef MATSAR_VECTORUTIL_HPP
#define MATSAR_VECTORUTIL_HPP

#include "UtilHeadCheck.hpp"

#ifndef MATSAR_EXCEPTIONUTIL_HPP

#include "ExceptionUtil.hpp"

#endif //MATSAR_EXCEPTIONUTIL_HPP
#ifndef MATSAR_TYPEUTIL_HPP

#include "TypeUtil.hpp"

#endif//MATSAR_TYPEUTIL_HPP

#include <vector>
#include <string>
#include <algorithm>

using std::vector;
using std::initializer_list;
using std::string;

using namespace matsar::Util::Exception;

//#if (__cplusplus < 201703L)
//
//// declare in C++17
//namespace std {
//    template<class E>
//    [[nodiscard]] constexpr bool empty(std::initializer_list<E> il) noexcept {
//        return il.size() == 0;
//    }
//}
//
//#endif

// Hand-laid-out code-generation macros below; keep their exact layout.
// clang-format off
#define CHECK_LIST_FIX_MAKER(LIST_n, LIST_n_1) \
template<class dType>\
bool checkListFix(LIST_n list_n) {\
    if(list_n.size() == 0) return true;\
    size_t size = list_n.begin()->size();\
    for(const LIST_n_1 &list_n_1:list_n)\
        if(checkListFix(list_n_1, size))\
            return false;\
    return true;\
}\
/** Check if the layer corresponds to size and recursively determine the internal */\
template<class dType>\
bool checkListFix(LIST_n list_n, size_t size) {\
    if (size == 0)\
        return list_n.size() == 0;\
    if (list_n.size() != size)\
        return false;\
    return checkListFix(list_n);\
}

#define UNINIT_LIST_COPY_MAKER(LIST_n, LIST_n_1) \
template<class dType>\
size_t uninitListCopy(LIST_n list, dType *p) {\
    dType *p0 = p;\
    for (LIST_n_1 l: list){\
        p+=uninitListCopy(l, p);\
    }\
    return p-p0;\
}
#define GET_FIX_LIST_SIZE_MAKER(LIST_n) \
template<class dType>\
size_t getFixListSize(LIST_n list) {\
    return list.size()==0 ? 0 : list.size() * getFixListSize(*list.begin());\
}

#define GET_DIM_SIZE_MAKER(LIST_n) \
template<class dType>\
vector<int64> getDimSize(LIST_n list_n) {\
    vector<int64> dim = getDimSize(*list_n.begin());\
    dim.push_back((int64)list_n.size());\
    return dim;\
}

#define CHECK_LIST_FIX_MAKER_N \
    CHECK_LIST_FIX_MAKER(LIST##2, LIST##1)\
    CHECK_LIST_FIX_MAKER(LIST##3, LIST##2)\
    CHECK_LIST_FIX_MAKER(LIST##4, LIST##3)\
    CHECK_LIST_FIX_MAKER(LIST##5, LIST##4)\
    CHECK_LIST_FIX_MAKER(LIST##6, LIST##5)\
    CHECK_LIST_FIX_MAKER(LIST##7, LIST##6)\
    CHECK_LIST_FIX_MAKER(LIST##8, LIST##7)

#define UNINIT_LIST_COPY_MAKER_N \
    UNINIT_LIST_COPY_MAKER(LIST##2, LIST##1)\
    UNINIT_LIST_COPY_MAKER(LIST##3, LIST##2)\
    UNINIT_LIST_COPY_MAKER(LIST##4, LIST##3)\
    UNINIT_LIST_COPY_MAKER(LIST##5, LIST##4)\
    UNINIT_LIST_COPY_MAKER(LIST##6, LIST##5)\
    UNINIT_LIST_COPY_MAKER(LIST##7, LIST##6)\
    UNINIT_LIST_COPY_MAKER(LIST##8, LIST##7)

#define GET_FIX_LIST_SIZE_MAKER_N \
    GET_FIX_LIST_SIZE_MAKER(LIST##2)\
    GET_FIX_LIST_SIZE_MAKER(LIST##3)\
    GET_FIX_LIST_SIZE_MAKER(LIST##4)\
    GET_FIX_LIST_SIZE_MAKER(LIST##5)\
    GET_FIX_LIST_SIZE_MAKER(LIST##6)\
    GET_FIX_LIST_SIZE_MAKER(LIST##7)\
    GET_FIX_LIST_SIZE_MAKER(LIST##8)

#define GET_DIM_SIZE_MAKER_N \
    GET_DIM_SIZE_MAKER(LIST##2)\
    GET_DIM_SIZE_MAKER(LIST##3)\
    GET_DIM_SIZE_MAKER(LIST##4)\
    GET_DIM_SIZE_MAKER(LIST##5)\
    GET_DIM_SIZE_MAKER(LIST##6)\
    GET_DIM_SIZE_MAKER(LIST##7)\
    GET_DIM_SIZE_MAKER(LIST##8)

#define LIST_FUN_MAKER_N \
namespace matsar{namespace Util{namespace VectorUtil {\
    CHECK_LIST_FIX_MAKER_N\
    UNINIT_LIST_COPY_MAKER_N\
    GET_FIX_LIST_SIZE_MAKER_N\
    GET_DIM_SIZE_MAKER_N \
    }}}


#define CHECK_VEC_FIX_MAKER(VEC_n, VEC_n_1) \
template<class dType>\
bool checkVecFix(VEC_n vec_n) {\
    if(vec_n.empty()) return true;\
    size_t size = vec_n.begin()->size();\
    for(const VEC_n_1 &vec_n_1:vec_n)\
        if(checkVecFix(vec_n_1, size))\
            return false;\
    return true;\
}\
/** Check if the layer corresponds to size and recursively determine the internal */\
template<class dType>\
bool checkVecFix(VEC_n vec_n, size_t size) {\
    if (size == 0)\
        return vec_n.empty();\
    if (vec_n.size() != size)\
        return false;\
    return checkVecFix(vec_n);\
}

#define VEC_COPY_MAKER(VEC_n, VEC_n_1) \
template<class dType>\
size_t VecCopy(VEC_n vec, dType *p) {\
    dType *p0 = p;\
    for (VEC_n_1 l: vec){\
        p+=VecCopy(l, p);\
    }\
    return p-p0;\
}
#define GET_FIX_VEC_SIZE_MAKER(VEC_n) \
template<class dType>\
size_t getFixVecSize(VEC_n vec) {\
    return vec.empty() ? 0 : vec.size() * getFixVecSize(*vec.begin());\
}

#define GET_VEC_DIM_SIZE_MAKER(VEC_n) \
template<class dType>\
vector<int64> getVecDimSize(VEC_n vec_n) {\
    vector<int64> dim = getVecDimSize(*vec_n.begin());\
    dim.push_back((int64)vec_n.size());\
    return dim;\
}

#define CHECK_VEC_FIX_MAKER_N \
    CHECK_VEC_FIX_MAKER(VEC##2, VEC##1)\
    CHECK_VEC_FIX_MAKER(VEC##3, VEC##2)\
    CHECK_VEC_FIX_MAKER(VEC##4, VEC##3)\
    CHECK_VEC_FIX_MAKER(VEC##5, VEC##4)\
    CHECK_VEC_FIX_MAKER(VEC##6, VEC##5)\
    CHECK_VEC_FIX_MAKER(VEC##7, VEC##6)\
    CHECK_VEC_FIX_MAKER(VEC##8, VEC##7)

#define VEC_COPY_MAKER_N \
    VEC_COPY_MAKER(VEC##2, VEC##1)\
    VEC_COPY_MAKER(VEC##3, VEC##2)\
    VEC_COPY_MAKER(VEC##4, VEC##3)\
    VEC_COPY_MAKER(VEC##5, VEC##4)\
    VEC_COPY_MAKER(VEC##6, VEC##5)\
    VEC_COPY_MAKER(VEC##7, VEC##6)\
    VEC_COPY_MAKER(VEC##8, VEC##7)

#define GET_FIX_VEC_SIZE_MAKER_N \
    GET_FIX_VEC_SIZE_MAKER(VEC##2)\
    GET_FIX_VEC_SIZE_MAKER(VEC##3)\
    GET_FIX_VEC_SIZE_MAKER(VEC##4)\
    GET_FIX_VEC_SIZE_MAKER(VEC##5)\
    GET_FIX_VEC_SIZE_MAKER(VEC##6)\
    GET_FIX_VEC_SIZE_MAKER(VEC##7)\
    GET_FIX_VEC_SIZE_MAKER(VEC##8)

#define GET_VEC_DIM_SIZE_MAKER_N \
    GET_VEC_DIM_SIZE_MAKER(VEC##2)\
    GET_VEC_DIM_SIZE_MAKER(VEC##3)\
    GET_VEC_DIM_SIZE_MAKER(VEC##4)\
    GET_VEC_DIM_SIZE_MAKER(VEC##5)\
    GET_VEC_DIM_SIZE_MAKER(VEC##6)\
    GET_VEC_DIM_SIZE_MAKER(VEC##7)\
    GET_VEC_DIM_SIZE_MAKER(VEC##8)

#define VEC_FUN_MAKER_N \
namespace matsar{namespace Util{namespace VectorUtil {\
    CHECK_VEC_FIX_MAKER_N\
    VEC_COPY_MAKER_N\
    GET_FIX_VEC_SIZE_MAKER_N\
    GET_VEC_DIM_SIZE_MAKER_N \
    }}}
// clang-format on

namespace matsar {
    namespace Util {
        namespace VectorUtil {
//            auto Range(int64 start, int64 end, int64 step) -> vector<int64> {
//                if (start >= end - step)
//                    return {};
//                auto r_size = (end - start) / step;
//                if ((end - start) % step != 0) {
//                    r_size++;
//                }
//                vector<int64> r(r_size);
//                int64 value = start;
//                for (decltype(r_size) i = 0; i < r_size; ++i) {
//                    r[i] = value;
//                    value += step;
//                }
//                return r;
//            }
//
//            auto Range(int64 start, int64 end) -> vector<int64> {
//                return Range(start, end, 1);
//            }
//
//            auto Range(int64 end) -> vector<int64> {
//                return Range(0, end, 1);
//            }

            template<IntType IT = int64 CheckInt(IT)>
            auto Range(const IT &start, const IT &end, const IT &step) -> vector<IT> {
                if (start >= end - step)
                    return {};
                auto r_size = (end - start) / step;
                if ((end - start) % step != 0) {
                    r_size++;
                }
                vector<IT> r(r_size);
                IT value = start;
                for (decltype(r_size) i = 0; i < r_size; ++i) {
                    r[i] = value;
                    value += step;
                }
                return r;
            }

            template<IntType IT = int64 CheckInt(IT)>
            auto Range(const IT &start, const IT &end) -> vector<IT> {
                return Range(start, end, IT(1));
            }

            template<IntType IT = int64 CheckInt(IT)>
            auto Range(const IT &end) -> vector<IT> {
                return Range(IT(0), end, IT(1));
            }

            inline auto fRange(float64 start, float64 end, float64 step) -> vector<float64> {
                if (start >= end - step)
                    return {};
                auto r_size = vector<float64>::size_type(std::ceil((end - start) / step));
                vector<float64> r(r_size);
                float64 value = start;
                for (decltype(r_size) i = 0; i < r_size; ++i) {
                    r[i] = value;
                    value += step;
                }
                return r;
            }

            inline auto fRange(float64 start, float64 end) -> vector<float64> {
                return fRange(start, end, 1);
            }

            inline auto fRange(float64 end) -> vector<float64> {
                return fRange(0, end, 1);
            }
        }

        namespace VectorUtil {
            template<class dType>
            constexpr auto checkListFix([[maybe_unused]] initializer_list<dType> list1) -> bool {
                return true;
            }

            /** Check whether this level matches the given size, then recurse into the inner levels. */
            template<class dType>
            auto checkListFix(initializer_list<dType> list, size_t size) -> bool {
                return list.size() == size;
            }

            template<class dType>
            auto uninitListCopy(initializer_list<dType> list, dType *p) -> size_t {
                std::uninitialized_copy_n(list.begin(), list.size(), p);
                return list.size();
            }

            template<class dType>
            auto getFixListSize(initializer_list<dType> list) -> size_t {
                return list.size();
            }

            template<class dType>
            auto getDimSize(initializer_list<dType> list) -> vector<int64> {
                return vector<int64>({(int64) list.size()});
            }

        }

        namespace VectorUtil {
            template<class dType>
            constexpr auto checkVecFix([[maybe_unused]] vector<dType> vec1) -> bool {
                return true;
            }

            /** Check whether this level matches the given size, then recurse into the inner levels. */
            template<class dType>
            auto checkVecFix(vector<dType> vec, size_t size) -> bool {
                return vec.size() == size;
            }

            template<class dType>
            auto VecCopy(vector<dType> vec, dType *p) -> size_t {
                std::uninitialized_copy_n(vec.begin(), vec.size(), p);
                return vec.size();
            }

            template<class dType>
            auto getFixVecSize(vector<dType> vec) -> size_t {
                return vec.size();
            }

            template<class dType>
            auto getVecDimSize(vector<dType> vec) -> vector<int64> {
                return vector<int64>({(int64) vec.size()});
            }

        }

        namespace VectorUtil {
            template<class T>
            auto cut(const vector<T> &v,
                     typename vector<T>::size_type begin,
                     typename vector<T>::size_type end) -> vector<T> {
                return vector<T>(v.begin() + begin, v.begin() + end);
            }

            template<class T>
            auto cut(const vector<T> &v,
                     typename vector<T>::size_type begin) -> vector<T> {
                return vector<T>(v.begin() + begin, v.end());
            }
        }

        namespace VectorUtil {
            template<class T>
            auto inRange(const vector<T> &v, typename vector<T>::size_type index) -> bool {
                return index >= 0 && index < v.size();
            }

            template<typename T>
            auto(mul)(const vector<T> &v) -> T {
                T s = 1;
                for (const auto &i: v)
                    s *= i;
                return s;
            }

            template<typename T>
            auto abs(const vector<T> &v) -> vector<T> {
                vector<T> r = v;
                for (auto &i: r)
                    i = std::abs(i);
                return r;
            }

            /**
             * times from [begin:end)
             */
            template<typename T>
            auto(mul)(const vector<T> &v,
                      const typename vector<T>::size_type begin, const typename vector<T>::size_type end) -> T {
                if (!inRange(v, begin))
                    throw_out_of_range_error(
                            "mul: begin = " + std::to_string(begin) + " out of vector Range = " +
                                    std::to_string(v.size()),
                            true);
                if (begin == end) return 1;

                if (!inRange(v, end - 1))
                    throw_out_of_range_error(
                            "mul: end = " + std::to_string(end) + " out of vector Range = " + std::to_string(v.size()),
                            true);
                if (begin > end)
                    throw_out_of_range_error(
                            "mul: begin index should be lower than end index (begin = " + std::to_string(begin) +
                                    " end = " +
                                    std::to_string(end),
                            true);
                T s = 1;
                for (auto i = v.begin() + begin; i != v.begin() + end; ++i)
                    s *= *i;
                return s;
            }

            //    /**
            //     * Excludes the element at 'end', i.e. multiplies over [begin:end-1]
            //     */
            //    template<Calculable T, class ITER1, class ITER2>
            //    T (mul)(ITER1 begin, const ITER2 &end) {
            //        if (begin > end)
            //            throw std::range_error("mul: begin iter should be lower than end iter");
            //        T s = 1;
            //        while (begin != end)
            //            s *= *(begin++);
            //        return s;
            //    }

            template<typename T>
            auto(sum)(const vector<T> &v) -> T {
                T s = 0;
                for (const auto &i: v)
                    s += i;
                return s;
            }

            /**
             * sum from [begin:end-1]
             */
            template<typename T>
            auto(sum)(const vector<T> &v,
                      const typename vector<T>::size_type begin, const typename vector<T>::size_type end) -> T {
                if (!inRange(v, begin))
                    throw_out_of_range_error(
                            "sum: begin =" + std::to_string(begin) + " out of vector Range size =" +
                                    to_string(v.size()),
                            true);
                if (!inRange(v, end - 1))
                    throw_out_of_range_error(
                            "sum: end =" + std::to_string(begin) + " out of vector Range size =" + to_string(v.size()),
                            true);
                if (begin > end)
                    throw_out_of_range_error(
                            "sum: begin index should be lower than end index (begin = " + std::to_string(begin) +
                                    " end = " +
                                    std::to_string(end),
                            true);
                T s = 0;
                for (auto i = v.begin() + begin; i != v.begin() + end; ++i)
                    s += *i;
                return s;
            }

//            /**
//             * sum from [begin:end-1]
//             */
//            template<Calculable T, class ITER1, class ITER2>
//            auto (sum)(ITER1 begin, const ITER2 &end) -> T {
//                if (begin > end)
//                    throw std::range_error("sum: begin iter should be lower than end iter");
//                T s = 0;
//                while (begin != end)
//                    s += *(begin++);
//                return s;
//            }

            template<class T>
            auto(max)(const vector<T> &v) -> T {
                if (v.empty()) return 0;
                T s = v[0];
                for (const auto &ele: v) {
                    if (ele > s)
                        s = ele;
                }
                return s;
            }

            template<class T, class Pr>
            auto(max)(const vector<T> &v, Pr pred) -> T {
                if (v.empty()) return 0;
                T s = v[0];
                for (const auto &ele: v) {
                    if (pred(ele, s))
                        s = ele;
                }
                return s;
            }

            template<class T>
            auto(min)(const vector<T> &v) -> T {
                if (v.empty()) return 0;
                T s = v[0];
                for (const auto &ele: v) {
                    if (ele < s)
                        s = ele;
                }
                return s;
            }

            template<class T, class Pr>
            auto(min)(const vector<T> &v, Pr pred) -> T {
                if (v.empty()) return 0;
                T s = v[0];
                for (const auto &ele: v) {
                    if (pred(ele, s))
                        s = ele;
                }
                return s;
            }

            /**
             * alin the vector according to the length.
             * @param vec &vector with reference
             * @param length target length, add num if less, remove if more
             * @param num default number to add
             */
            template<typename T>
            void alignNum(vector<T> &vec, const int64 length, const T num = T()) {
                if (length < 0 || length == vec.size())
                    return;
                if (vec.size() > length) {
                    vec.erase(vec.begin() + length, vec.end());
                } else {
                    vec.insert(vec.end(), length - vec.size(), num);
                }
            }

            /**
             * aline 2dim vector
             * @param vec 2dim vector
             * @param targetShape target 2d shape, add num if less, remove if more
             * @param num default number to add
             */
            template<Calculable T>
            void alignNum2d(vector<vector<T>> &vec, vector<int64> targetShape = {-1, -1}, T num = T()) {
                for (int i = 0; i < 2 - targetShape.size(); ++i)
                    targetShape.push_back(-1);

                //handle dim1
                if (targetShape[0] < 0) {
                    for (vector<T> &v: vec) {
                        targetShape[0] = std::max(targetShape[0], v.size());
                    }
                }
                for (auto &v: vec) {
                    matsar::Util::VectorUtil::alignNum(v, targetShape[0], num);
                }
                //handel dim2
                if (targetShape[1] >= 0) {
                    if (vec.size() > targetShape[1]) {
                        vec.erase(vec.begin() + targetShape[1], vec.end());
                    } else {
//                        for (int i = 0; i < targetShape[1] - vec.size(); ++i) {
//                            vec.emplace_back({targetShape[0], num});
//                        }
                        vec.insert(vec.end(), targetShape[1] - vec.size(), vector<T>(targetShape[0], num));
                    }
                }
            }

            /**
             * C++ style to_string
             */
            template<class T>
            auto to_string(const vector<T> &v, const string &segment = ", ",
                           const string &left_bracket = "{", const string &right_bracket = "}") -> string {
                stringstream ss;
                ss << left_bracket;
                auto iter = v.begin();
                while (iter != v.end()) {
                    ss << *iter;
                    if (++iter != v.end()) {
                        ss << segment;
                    }
                }
                ss << right_bracket;
                return ss.str();
            }

            /**
             * Java style toString
             */
            template<class T>
            inline static auto toString(const vector<T> &v, const string &segment = ", ",
                                        const string &left_bracket = "{", const string &right_bracket = "}") -> string {
                return to_string(v, segment, left_bracket, right_bracket);
            }

        }
    }
}

LIST_FUN_MAKER_N

VEC_FUN_MAKER_N

#endif //MATSAR_VECTORUTIL_HPP
