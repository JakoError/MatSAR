//
// Created by JakoError on 2022/7/10.
//
///对矩阵符号函数的扩展
#ifndef MATSAR_MATRIXOPERATION_HPP
#define MATSAR_MATRIXOPERATION_HPP

#include <numeric>
#include <algorithm>
#include <unordered_map>

#include <cblas.h>

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

#ifndef MATSAR_TRIGONOMETRY_HPP

#include "Trigonometry.hpp"

#endif

#ifndef MATSAR_LINEARALGEBRA_HPP

#include "LinearAlgebra.hpp"

#endif

namespace MatSAR {
    ///complex
    namespace Math {
        template<typename dType>
        inline auto real(const Matrix<dType> &m) -> decltype(m.real()) {
            return m.real();
        }

        template<typename dType>
        inline auto image(const Matrix<dType> &m) -> decltype(m.image()) {
            return m.image();
        }

        template<typename dType>
        inline auto conj(const Matrix<dType> &m) -> decltype(m.conj()) {
            return m.conj();
        }
    }

    namespace Math {
        template<typename dType, typename TargetType=decltype(MatSAR::ceil(std::declval<dType>()))>
        static auto ceil(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::ceil);
        }

        template<typename dType, typename TargetType=decltype(MatSAR::floor(std::declval<dType>()))>
        static auto floor(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::floor);
        }

        template<typename dType, typename TargetType=decltype(MatSAR::round(std::declval<dType>()))>
        static auto round(const Matrix<dType> &m, int n = 0) -> Matrix<TargetType> {
            if (n == 0) {
                return apply_function<dType, TargetType>(m, MatSAR::round);
            } else {
                return apply_function<dType, TargetType>(m * pow(10, n), MatSAR::round) * pow(0.1, n);
            }
        }

        template<typename dType, typename TargetType=decltype(MatSAR::fix(std::declval<dType>()))>
        static auto fix(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::fix);
        }

        template<IntType IT CheckInt(IT)>
        static auto isPrime(const Matrix<IT> &m) -> Matrix<bool> {
            return apply_function<IT, bool>(m, MatSAR::isPrime);
        }

        template<IntType IT1, IntType IT2 CheckInt(IT1)CheckInt(IT2)>
        static auto numden(const Matrix<IT1> &m1, const Matrix<IT2> &m2) -> vector<Matrix<int64>> {
            Matrix<int64> r1(m1);
            Matrix<int64> r2(m2);
            Matrix<int64>::broadcast(r1, r2);
            for (auto iter1 = r1.begin(), iter2 = r2.begin(); iter1 != r1.end(); ++iter1, ++iter2) {
                auto values = MatSAR::numden(*iter1, *iter2);
                *iter1 = values[0];
                *iter2 = values[1];
            }
            return {r1, r2};
        }
    }

    ///special function
    namespace Math {
        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Gcd(leftT *left, rightT *right, msize_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = MatSAR::gcd(*left++, *right++);
        }

        template<typename IT1, typename IT2, typename TargetType=decltype(gcd(declval<IT1>(), declval<IT2>()))>
        auto gcd(const Matrix<IT1> &m1, const Matrix<IT2> &m2) -> Matrix<TargetType> {
            if (m1.getShape() == m2.getShape()) {
                Matrix<TargetType> r(m1.getShape(), 0., m1.getData());
                Gcd(m1.getData(), m2.getData(), r.getSize(), r.getData());
                return r;
            } else {
                auto b_m1 = m1;
                auto b_m2 = m2;
                Matrix<TargetType>::broadcast(b_m1, b_m2);
                Matrix<TargetType> r(b_m1.getShape(), 0., m1.getData());
                Gcd(b_m1.getData(), b_m2.getData(), r.getSize(), r.getData());
                return r;
            }
        }

        ///FIXME Optimizable factorial algorithm using array to record data
        template<typename IT, typename TargetType=float_max>
        auto factorial(const Matrix<IT> &m) -> Matrix<TargetType> {
            Matrix<TargetType> r(m);
            for (auto &i: r)
                i = MatSAR::factorial<TargetType>(i);
            return r;
        }

        ///NO CHECK!
        template<typename leftT, typename rightT, class DstT>
        static void Nchoosek(leftT *left, rightT *right, msize_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = MatSAR::nchoosek<leftT, rightT, DstT>(*left++, *right++);
        }

        template<typename IT1, typename IT2, typename TargetType=float_max>
        auto nchoosek(const Matrix<IT1> &m1, const Matrix<IT2> &m2) -> Matrix<TargetType> {
            if (m1.getShape() == m2.getShape()) {
                Matrix<TargetType> r(m1.getShape(), 0., m1.getData());
                Nchoosek(m1.getData(), m2.getData(), r.getSize(), r.getData());
                return r;
            } else {
                auto b_m1 = m1;
                auto b_m2 = m2;
                Matrix<TargetType>::broadcast(b_m1, b_m2);
                Matrix<TargetType> r(b_m1.getShape(), 0., m1.getData());
                Nchoosek(b_m1.getData(), b_m2.getData(), r.getSize(), r.getData());
                return r;
            }
        }
    }

    ///logx exp
    namespace Math {
        template<typename dType, typename TargetType = decltype(MatSAR::log(std::declval<dType>()))>
        auto log(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::log);
        }

        template<typename dType, typename TargetType = decltype(MatSAR::log(std::declval<dType>()))>
        auto logm(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::log);
        }

        template<typename dType, typename TargetType = decltype(MatSAR::log(std::declval<dType>()))>
        auto ln(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::log);
        }

        template<typename dType, typename TargetType = decltype(MatSAR::log2(std::declval<dType>()))>
        auto log2(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::log2);
        }

        template<typename dType, typename TargetType = decltype(MatSAR::log10(std::declval<dType>()))>
        auto log10(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::log10);
        }

        template<typename dType, typename TargetType = decltype(MatSAR::exp(std::declval<dType>()))>
        auto exp(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::exp);
        }

        template<typename dType, typename TargetType = decltype(MatSAR::exp(std::declval<dType>()))>
        auto expm(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::exp);
        }

        template<typename dType, typename TargetType = decltype(MatSAR::exp2(std::declval<dType>()))>
        auto exp2(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::exp2);
        }
    }
    ///gamma
    namespace Math {
        template<typename dType, typename TargetType = decltype(MatSAR::gamma(std::declval<dType>()))>
        auto gamma(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::gamma);
        }

        template<typename dType, typename TargetType = decltype(MatSAR::gammaln(std::declval<dType>()))>
        auto gammaln(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::gammaln);
        }
    }

    ///mpower sqrt
    namespace Math {
        template<typename dType, typename TargetType = decltype(MatSAR::sqrt(std::declval<dType>()))>
        auto sqrt(const Matrix<dType> &m) -> Matrix<TargetType> {
            return apply_function<dType, TargetType>(m, MatSAR::sqrt);
        }

        template<typename lT, typename rT>
        inline auto mpower(const Matrix<lT> &left, const Matrix<rT> &right) -> decltype(left ^ right) {
            return left ^ right;
        }

        template<typename lT, typename rT>
        inline auto mpower(const Matrix<lT> &left, const rT &right) -> decltype(left ^ right) {
            return left ^ right;
        }

        template<typename lT, typename rT>
        inline auto mpower(const lT &left, const Matrix<rT> &right) -> decltype(left ^ right) {
            return left ^ right;
        }
    }

    ///matrix alter
    namespace Math {
        template<typename dType>
        inline auto T(const Matrix<dType> &m) -> Matrix<dType> {
            return m.T();
        }

        template<typename T, typename TargetType=decltype(MatSAR::abs(std::declval<T>()))>
        auto abs(const Matrix<T> &m) -> Matrix<TargetType> {
            return apply_function(m, MatSAR::abs);
        }

//        template<Calculable dType>
//        auto det(const Matrix<dType> &m) -> decltype(m.det()) {
//            return m.det();
//        }

//        template<Calculable dType>
//        auto inv(const Matrix<dType> &m) -> decltype(m.inv()) {
//            return m.inv();
//        }

        template<typename dType>
        inline auto cat(const vector<Matrix<dType>> &mv, const int64 &dim = 1) -> Matrix<dType> {
            return {mv, dim};
        }

        template<typename dType>
        inline auto cat(const initializer_list<Matrix<dType>> &mv, const int64 &dim = 1) -> Matrix<dType> {
            return {mv, dim};
        }
    }

    ///check matrix (iss)
    namespace Math {
        template<typename dType>
        inline auto isSquare(const Matrix<dType> &m) -> bool {
            return m.isSquare();
        }

        template<typename dType>
        inline auto isDiag(const Matrix<dType> &m) -> bool {
            return m.isDiag();
        }

        template<typename dType>
        inline auto isEmpty(const Matrix<dType> &m) -> bool {
            return m.isEmpty();
        }
    }

    ///permute
    namespace Math {
        /**
         * permute matrix according to dimorder
         */
        template<typename dType>
        auto permute(const Matrix<dType> &m, mshape_type dimorder) -> Matrix<dType> {
            if (dimorder.size() != m.getDims()) {
                throw_value_error("permute: dimorder needs to have the same length as the matrix dimensions");
                return {};
            }
            //New Matrix Shape
            vector<bool> pick_check(dimorder.size(), false);
            auto r_shape = m.getShape();
            for (size_t i = 0; i < dimorder.size(); ++i) {
                if (dimorder[i] <= 0 || dimorder[i] > m.getDims()) {
                    throw_value_error("permute: The number in dimorder must be from 1 to n");
                }
                dimorder[i] = dimorder[i] - 1;//trans numeric index to array index
                if (pick_check[dimorder[i]]) {
                    throw_value_error("permute: The numbers in dimorder are repeated");
                }
                pick_check[dimorder[i]] = false;
                if (i != dimorder[i]) {
                    r_shape[i] = m.getShape()[dimorder[i]];
                }
            }
            //New Matrix Data Acquisition
            //312->231
            auto idimorder(dimorder);
            for (size_t i = 0; i < dimorder.size(); ++i) {
                idimorder[dimorder[i]] = msize_type(i);
            }

            Matrix<dType> r(r_shape, 0.);
            mshape_type m_picker(m.getDims(), 0);
            mshape_type r_picker(m.getDims(), 0);

            for (auto &i: m) {
                r(r_picker) = i;
                for (size_t j = 0; j < m_picker.size(); ++j) {
                    m_picker[j]++;
                    if (m_picker[j] == m.getShape()[j]) {
                        m_picker[j] = 0;
                        r_picker[idimorder[j]] = m_picker[j];
                        continue;
                    }
                    r_picker[idimorder[j]] = m_picker[j];
                    break;
                }
            }
            return r;
        }

        /**
         * Inverse swap the dimensions of the matrix using dimorder as the original order
         */
        template<typename dType>
        auto ipermute(const Matrix<dType> &m, mshape_type dimorder) -> Matrix<dType> {
            auto idimorder(dimorder);
            for (size_t i = 0; i < dimorder.size(); ++i) {
                //trans numeric index to array index -1 +1
                idimorder[dimorder[i] - 1] = msize_type(i + 1);
            }
            return permute(m, idimorder);
        }
    }

    namespace Math {
        /**
         * nnz, The total number of nonzero elements in the matrix
         */
        template<typename dType>
        auto nnz(const Matrix<dType> &m) -> msize_type {
            return sum<bool, msize_type>(m != 0, none)(0);
        }

        /**
         * nonzeros, Non-zero elements of the matrix
         * @author:jie
         */
        template<typename dType>
        auto nonzeros(const Matrix<dType> &m) -> Matrix<dType> {
            auto r = zeros<dType>({nnz(m)});
            auto iter_r = r.begin();
            for (const auto &i: m) {
                if (i != 0) {
                    *iter_r = i;
                    iter_r++;
                }
            }
            return r;
        }
    }
}


#endif //MATSAR_MATRIXOPERATION_HPP