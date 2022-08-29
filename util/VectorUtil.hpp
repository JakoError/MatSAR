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

using namespace MatSAR::Util::Exception;

#define CHECK_LIST_FIX_MAKER(LIST_n, LIST_n_1) \
template<class dType>\
constexpr bool checkListFix(LIST_n list_n) {\
    if(std::empty(list_n)) return true;\
    size_t size = list_n.begin()->size();\
    for(const LIST_n_1 &list_n_1:list_n)\
        if(checkListFix(list_n_1, size))\
            return false;\
    return true;\
}\
/** 检查该层是不是对应size大小,并且递归地判断内部 */\
template<class dType>\
constexpr bool checkListFix(LIST_n list_n, size_t size) {\
    if (size == 0)\
        return std::empty(list_n);\
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
    return std::empty(list) ? 0 : list.size() * getFixListSize(*list.begin());\
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
namespace MatSAR::Util::VectorUtil {\
    CHECK_LIST_FIX_MAKER_N\
    UNINIT_LIST_COPY_MAKER_N\
    GET_FIX_LIST_SIZE_MAKER_N\
    GET_DIM_SIZE_MAKER_N \
    }


#define CHECK_VEC_FIX_MAKER(VEC_n, VEC_n_1) \
template<class dType>\
constexpr bool checkVecFix(VEC_n vec_n) {\
    if(vec_n.empty()) return true;\
    size_t size = vec_n.begin()->size();\
    for(const VEC_n_1 &vec_n_1:vec_n)\
        if(checkVecFix(vec_n_1, size))\
            return false;\
    return true;\
}\
/** 检查该层是不是对应size大小,并且递归地判断内部 */\
template<class dType>\
constexpr bool checkVecFix(VEC_n vec_n, size_t size) {\
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
namespace MatSAR::Util::VectorUtil {\
    CHECK_VEC_FIX_MAKER_N\
    VEC_COPY_MAKER_N\
    GET_FIX_VEC_SIZE_MAKER_N\
    GET_VEC_DIM_SIZE_MAKER_N \
    }


namespace MatSAR::Util {
    namespace VectorUtil {
        auto Range(int64 start, int64 end, int64 step) -> vector<int64> {
            if (start >= end - step)
                return {};
            vector<int64> r = {start};
            auto iter = r.rbegin();
            while (*iter < end - step) {
                r.push_back(*iter + step);
                iter = r.rbegin();
            }
            return r;
        }

        auto Range(int64 start, int64 end) -> vector<int64> {
            return Range(start, end, 1);
        }

        auto Range(int64 end) -> vector<int64> {
            return Range(0, end, 1);
        }

        auto fRange(float64 start, float64 end, float64 step) -> vector<float64> {
            if (start >= end - step)
                return {};
            vector<float64> r = {start};
            auto iter = r.rbegin();
            while (*iter <= end - step) {
                r.push_back(*iter + step);
                iter = r.rbegin();
            }
            return r;
        }

        auto fRange(float64 start, float64 end) -> vector<float64> {
            return fRange(start, end, 1);
        }

        auto fRange(float64 end) -> vector<float64> {
            return fRange(0, end, 1);
        }
    }
    namespace VectorUtil {
        template<class dType>
        constexpr bool checkListFix(initializer_list<dType> list1) {
            return true;
        }

        /** 检查该层是不是对应size大小,并且递归地判断内部 */
        template<class dType>
        constexpr bool checkListFix(initializer_list<dType> list, size_t size) {
            return list.size() == size;
        }

        template<class dType>
        size_t uninitListCopy(initializer_list<dType> list, dType *p) {
            std::uninitialized_copy_n(list.begin(), list.size(), p);
            return list.size();
        }

        template<class dType>
        size_t getFixListSize(initializer_list<dType> list) {
            return list.size();
        }

        template<class dType>
        vector<int64> getDimSize(initializer_list<dType> list) {
            return vector<int64>({(int64) list.size()});
        }

    }

    namespace VectorUtil {
        template<class dType>
        constexpr bool checkVecFix(vector<dType> vec1) {
            return true;
        }

        /** 检查该层是不是对应size大小,并且递归地判断内部 */
        template<class dType>
        constexpr bool checkVecFix(vector<dType> vec, size_t size) {
            return vec.size() == size;
        }

        template<class dType>
        size_t VecCopy(vector<dType> vec, dType *p) {
            std::uninitialized_copy_n(vec.begin(), vec.size(), p);
            return vec.size();
        }

        template<class dType>
        size_t getFixVecSize(vector<dType> vec) {
            return vec.size();
        }

        template<class dType>
        vector<int64> getVecDimSize(vector<dType> vec) {
            return vector<int64>({(int64) vec.size()});
        }

    }

    namespace VectorUtil {
        template<class T>
        bool inRange(const vector<T> &v, typename vector<T>::size_type index) {
            return index >= 0 && index < v.size();
        }

        template<Calculable T>
        T (mul)(const vector<T> &v) {
            T s = 1;
            for (const int &i: v)
                s *= i;
            return s;
        }

        template<Calculable T>
        vector<T> abs(const vector<T> &v) {
            vector<T> r = v;
            for (auto &i: r)
                i = std::abs(i);
            return r;
        }

        /**
         * 不包含end位置的元素即：[begin:end)相乘
         */
        template<Calculable T>
        T (mul)(const vector<T> &v, typename vector<T>::size_type begin, typename vector<T>::size_type end) {
            if (!inRange(v, begin))
                throw_out_of_range_error(
                        "mul: begin = " + std::to_string(begin) + " out of vector Range = " + std::to_string(v.size()),
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
                        std::to_string(end), true);
            T s = 1;
            for (auto i = v.begin() + begin; i != v.begin() + end; ++i)
                s *= *i;
            return s;
        }

//    /**
//     * 不包含end位置的元素即：[begin:end-1]相乘
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

        template<Calculable T>
        T (sum)(const vector<T> &v) {
            T s = 0;
            for (const int &i: v)
                s += i;
            return s;
        }

        /**
         * 不包含end位置的元素即：[begin:end-1]相加
         */
        template<Calculable T>
        T (sum)(const vector<T> &v, const typename vector<T>::size_type &begin,
                const typename vector<T>::size_type &end) {
            if (!inRange(v, begin))
                throw std::out_of_range(
                        "sum: begin =" + std::to_string(begin) + " out of vector Range size =" + to_string(v.size()));
            if (!inRange(v, end - 1))
                throw std::out_of_range(
                        "sum: end =" + std::to_string(begin) + " out of vector Range size =" + to_string(v.size()));
            if (begin > end)
                throw std::range_error("sum: begin index should be lower than end index");
            T s = 0;
            for (auto i = v.begin() + begin; i != v.begin() + end; ++i)
                s += *i;
            return s;
        }

        /**
         * 不包含end位置的元素即：[begin:end-1]相加
         */
        template<Calculable T, class ITER1, class ITER2>
        T (sum)(ITER1 begin, const ITER2 &end) {
            if (begin > end)
                throw std::range_error("mul: begin iter should be lower than end iter");
            T s = 0;
            while (begin != end)
                s += *(begin++);
            return s;
        }

        template<class T>
        T (max)(vector<T> v) {
            if (v.empty()) return 0;
            T s = v[0];
            for (const auto &ele: v) {
                if (ele > s)
                    s = ele;
            }
            return s;
        }

        template<class T, class Pr>
        T (max)(vector<T> v, Pr pred) {
            if (v.empty()) return 0;
            T s = v[0];
            for (const auto &ele: v) {
                if (pred(ele, s))
                    s = ele;
            }
            return s;
        }

        template<class T>
        T (min)(vector<T> v) {
            if (v.empty()) return 0;
            T s = v[0];
            for (const auto &ele: v) {
                if (ele < s)
                    s = ele;
            }
            return s;
        }

        template<class T, class Pr>
        T (min)(vector<T> v, Pr pred) {
            if (v.empty()) return 0;
            T s = v[0];
            for (const auto &ele: v) {
                if (pred(ele, s))
                    s = ele;
            }
            return s;
        }

        template<Calculable T>
        void alignNum(vector<T> &vec, int64 length, T num = 0) {
            if (length < 0 || length == vec.size())
                return;
            auto size = vec.size();
            for (int64 i = 0; i < length - size; ++i)
                vec.push_back(num);
            vec.erase(vec.begin() + length, vec.end());
        }


        /**
         *
         * @tparam T
         * @param vec
         * @param targetShape
         * @param num
         */
        template<Calculable T>
        void alignNum2d(vector<vector<T>> &vec, vector<int> targetShape = {-1, -1}, T num = 0) {
            for (int i = 0; i < 2 - targetShape.size(); ++i)
                targetShape.push_back(-1);

            //handle dim1
            if (targetShape[0] < 0) {
                auto fun = [](vector<T> v) { return v.size(); };
                vector<typename vector<T>::size_type> sizes;
                for (auto &v: vec)
                    sizes.push_back(fun(v));
                auto max_size = targetShape[0] = max(sizes);
                vector<T> nums(max_size, num);
                for (typename vector<T>::size_type i = 0; i < vec.size(); ++i)
                    vec.at(i).insert(vec.at(i).end(), nums.begin(), nums.begin() + (max_size - sizes[i]));
            } else {
                for (auto &v: vec) {
                    for (int i = 0; i < targetShape[0] - (int) v.size(); ++i)
                        v.push_back(num);
                    v.erase(v.begin() + targetShape[0], v.end());
                }
            }
            //handel dim2
            if (targetShape[1] >= 0) {
                if (vec.size() > targetShape[1])
                    vec.erase(vec.begin() + targetShape[1], vec.end());
                else
                    for (int i = 0; i < targetShape[1] - vec.size(); ++i)
                        vec.push_back(vector<T>(targetShape[0], num));
            }
        }

        template<Calculable T>
        void alignZero2d(vector<vector<T>> &vec, vector<int> targetShape = {-1, -1}) { alignNum2d(vec, targetShape); }

        /**
         * C++ style vector
         */
        template<class T>
        string to_string(const vector<T> &v, const string &segment = ", ") {
            stringstream ss;
            ss << "{";
            auto iter = v.begin();
            while (iter != v.end()) {
                ss << *iter;
                if (++iter != v.end())
                    ss << segment;
            }
            ss << "}";
            return ss.str();
        }

        /**
         * C++ style vector
         */
        template<class T>
        string toString(const vector<T> &v, const string &segment = ", ") {
            return to_string(v, segment);
        }

    }
}

LIST_FUN_MAKER_N

VEC_FUN_MAKER_N

#endif //MATSAR_VECTORUTIL_HPP
