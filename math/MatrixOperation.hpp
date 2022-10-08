//
// Created by JakoError on 2022/7/10.
//
///对矩阵符号函数的扩展
#ifndef MATSAR_MATRIXOPERATION_HPP
#define MATSAR_MATRIXOPERATION_HPP

#include <numeric>

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

namespace MatSAR {
    ///complex
    namespace Math {
        template<Calculable dType>
        auto real(const Matrix<dType> &m) -> Matrix<float_max> {
            return m.real();
        }

        template<Calculable dType>
        auto image(const Matrix<dType> &m) -> Matrix<float_max> {
            return m.image();
        }

        template<Calculable dType>
        auto conj(const Matrix<dType> &m) -> Matrix<complex_max> {
            return m.conj();
        }
    }

    namespace Math {
        template<Calculable dType>
        static auto ceil(const Matrix<dType> &m) -> Matrix<dType> {
            Matrix<dType> r(m);
            for (auto &i: r)
                i = MatSAR::ceil(i);
            return r;
        }

        template<Calculable dType>
        static auto floor(const Matrix<dType> &m) -> Matrix<dType> {
            Matrix<dType> r(m);
            for (auto &i: r)
                i = MatSAR::floor(i);
            return r;
        }

        template<Calculable dType>
        static auto round(const Matrix<dType> &m) -> Matrix<dType> {
            Matrix<dType> r(m);
            for (auto &i: r)
                i = MatSAR::round(i);
            return r;
        }

        template<Calculable dType>
        static auto fix(const Matrix<dType> &m) -> Matrix<dType> {
            Matrix<dType> r(m);
            for (auto &i: r)
                i = MatSAR::fix(i);
            return r;
        }

        template<IntType IT>
        static auto isPrime(const Matrix<IT> &m) -> Matrix<int32> {
            Matrix<int32> r(m.getShape(), 0.,m.getData());
            auto iter_r = r.begin();
            for (auto iter = m.begin(); iter != m.end(); ++iter, ++iter_r)
                *iter_r = MatSAR::isPrime(*iter);
            return r;
        }

        template<IntType IT1, IntType IT2>
        static auto numden(const Matrix<IT1> &m1, const Matrix<IT2> &m2) -> tuple<Matrix<int64>, Matrix<int64>> {
            Matrix<int64> r1(m1);
            Matrix<int64> r2(m2);
            if (m1.getShape() != m2.getShape())
                Matrix<int64>::broadcast(r1, r2);
            for (auto iter1 = r1.begin(), iter2 = r2.begin(); iter1 != r1.end(); ++iter1, ++iter2)
                std::tie(*iter1, *iter2) = MatSAR::numden(*iter1, *iter2);
            return {r1, r2};
        }
    }

    ///special function
    namespace Math {
        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void Gcd(leftT *left, rightT *right, msize_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = std::gcd(*left++, *right++);
        }

        template<SIntType SIT1, SIntType SIT2>
        auto gcd(const Matrix<SIT1> &m1, const Matrix<SIT2> &m2) -> Matrix<int_max> {
            if (m1.getShape() == m2.getShape()) {
                Matrix<int_max> r(m1.getShape(), 0.,m1.getData());
                Gcd(m1.getData(), m2.getData(), r.getSize(), r.getData());
                return r;
            } else {
                auto b_m1 = m1;
                auto b_m2 = m2;
                Matrix<int_max>::broadcast(b_m1, b_m2);
                Matrix<int_max> r(b_m1.getShape(), 0.,m1.getData());
                Gcd(b_m1.getData(), b_m2.getData(), r.getSize(), r.getData());
                return r;
            }
        }

        template<UIntType UIT1, UIntType UIT2>
        auto gcd(const Matrix<UIT1> &m1, const Matrix<UIT2> &m2) -> Matrix<uint_max> {
            if (m1.getShape() == m2.getShape()) {
                Matrix<uint_max> r(m1.getShape(), 0.,m1.getData());
                Gcd(m1.getData(), m2.getData(), r.getSize(), r.getData());
                return r;
            } else {
                auto b_m1 = m1;
                auto b_m2 = m2;
                Matrix<uint_max>::broadcast(b_m1, b_m2);
                Matrix<uint_max> r(b_m1.getShape(), 0.,m1.getData());
                Gcd(b_m1.getData(), b_m2.getData(), r.getSize(), r.getData());
                return r;
            }
        }

        template<Calculable TargetType, RealType RT>
        auto factorial(RT i) -> TargetType {
            if (i < 0)
                throw_value_error("factorial阶乘不能求负");
            if (i != std::round(i))
                throw ValueError("fractorial需要整数");
            TargetType r = 1;
            while (i > 1)
                r *= i--;
            return r;
        }

        ///FIXME 可优化阶乘算法使用数组记录数据
        template<Calculable TargetType, IntType IT>
        auto factorial(const Matrix<IT> &m) -> Matrix<TargetType> {
            Matrix<TargetType> r(m);
            for (auto &i: r)
                i = factorial<TargetType>(i);
            return r;
        }

        template<RealType RT>
        auto factorial(const Matrix<RT> &m) -> Matrix<real_max> {
            return factorial<real_max>(m);
        }

        template<Calculable TargetType, RealType RT1, RealType RT2>
        auto nchoosek(RT1 n, RT2 k) -> TargetType {
            if (n != MatSAR::round(n) || k != MatSAR::round(k))
                return NAN;
            if (n < 0 || k < 0){
//                throw_value_error("nchoosek组合不能求负", false);
                return NAN;
            }
            if (k > n){
//                throw_value_error("nchoosek组合数k不能大于n", false);
                return NAN;
            }
            TargetType r = 1;
            for (RT2 i = 0; i < k; ++i)
                r *= n - i;
            for (RT2 j = 2; j <= k; ++j)
                r /= j;
            return r;
        }

        ///NO CHECK!
        template<Calculable leftT, Calculable rightT, class DstT>
        static void Nchoosek(leftT *left, rightT *right, msize_type n, DstT *dst) {
            while (n-- > 0)
                *dst++ = nchoosek<DstT>(*left++, *right++);
        }

        template<RealType TargetType, RealType RT1, RealType RT2>
        auto nchoosek(const Matrix<RT1> &m1, const Matrix<RT2> &m2) -> Matrix<TargetType> {
            if (m1.getShape() == m2.getShape()) {
                Matrix<TargetType> r(m1.getShape(), 0.,m1.getData());
                Nchoosek(m1.getData(), m2.getData(), r.getSize(), r.getData());
                return r;
            } else {
                auto b_m1 = m1;
                auto b_m2 = m2;
                Matrix<TargetType>::broadcast(b_m1, b_m2);
                Matrix<TargetType> r(b_m1.getShape(), 0.,m1.getData());
                Nchoosek(b_m1.getData(), b_m2.getData(), r.getSize(), r.getData());
                return r;
            }
        }

        template<RealType RT1, RealType RT2>
        auto nchoosek(const Matrix<RT1> &m1, const Matrix<RT2> &m2) -> Matrix<real_max> {
            return nchoosek<real_max>(m1, m2);
        }
    }

    ///logx exp
    namespace Math {
        template<RealType RT>
        auto reallog(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (Calculable auto &i: r) {
                if (i < 0)
                    throw_value_error("reallog需要运算数均大于0");
                i = std::log(i);
            }
            return r;
        }

        template<ComplexType CT>
        auto reallog([[maybe_unused]]const Matrix<CT> &m) -> Matrix<complex_max> {
            throw_value_error("reallog不支持复数矩阵");
            return {};
        }


        template<RealType RT>
        auto log(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (Calculable auto &i: r)
                i = std::log(i);
            return r;
        }

        template<ComplexType CT>
        auto log(const Matrix<CT> &m) -> Matrix<complex_max> {
            Matrix<complex_max> r(m);
            for (Calculable auto &i: r)
                i = std::log(i);
            return r;
        }

        template<RealType RT>
        auto logm(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (Calculable auto &i: r)
                i = std::log(i);
            return r;
        }

        template<ComplexType CT>
        auto logm(const Matrix<CT> &m) -> Matrix<complex_max> {
            Matrix<complex_max> r(m);
            for (Calculable auto &i: r)
                i = std::log(i);
            return r;
        }

        template<RealType RT>
        auto ln(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (Calculable auto &i: r)
                i = std::log(i);
            return r;
        }

        template<ComplexType CT>
        auto ln(const Matrix<CT> &m) -> Matrix<complex_max> {
            Matrix<complex_max> r(m);
            for (Calculable auto &i: r)
                i = std::log(i);
            return r;
        }

        template<RealType RT>
        auto log2(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (Calculable auto &i: r)
                i = std::log2(i);
            return r;
        }

        template<ComplexType CT>
        auto log2(const Matrix<CT> &m) -> Matrix<complex_max> {
            Matrix<complex_max> r(m);
            for (Calculable auto &i: r)
                i = MatSAR::log2(i);
            return r;
        }

        template<RealType RT>
        auto log10(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (Calculable auto &i: r)
                i = std::log10(i);
            return r;
        }

        template<ComplexType CT>
        auto log10(const Matrix<CT> &m) -> Matrix<complex_max> {
            Matrix<complex_max> r(m);
            for (Calculable auto &i: r)
                i = log10(i);
            return r;
        }

        template<RealType RT>
        auto exp(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (Calculable auto &i: r)
                i = std::exp(i);
            return r;
        }

        template<ComplexType CT>
        auto exp(const Matrix<CT> &m) -> Matrix<complex_max> {
            Matrix<complex_max> r(m);
            for (Calculable auto &i: r)
                i = std::exp(i);
            return r;
        }

        template<RealType RT>
        auto exp2(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (Calculable auto &i: r)
                i = std::exp2(i);
            return r;
        }

//        template<ComplexType CT>
//        auto exp2(const Matrix<CT> &m) -> Matrix<complex_max> {
//            Matrix<complex_max> r(m);
//            for (auto &i: r)
//                i = std::exp2(i);
//            return r;
//        }

        template<RealType RT>
        auto expm(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (Calculable auto &i: r)
                i = std::exp(i);
            return r;
        }

        template<ComplexType CT>
        auto expm(const Matrix<CT> &m) -> Matrix<complex_max> {
            Matrix<complex_max> r(m);
            for (Calculable auto &i: r)
                i = std::exp(i);
            return r;
        }
    }
    ///gamma
    namespace Math{
        template<RealType RT>
        auto gamma(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (Calculable auto &i: r)
                i = MatSAR::gamma(i);
            return r;
        }

        template<RealType RT>
        auto gammaln(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (Calculable auto &i: r)
                i = MatSAR::gammaln(i);
            return r;
        }
//        template<ComplexType CT>
//        auto gamma(const Matrix<CT> &m) -> Matrix<complex_max> {
//            Matrix<complex_max> r(m);
//            for (auto &i: r)
//                i = std::tgamma(i);
//            return r;
//        }
    }

    ///mpower sqrt
    namespace Math {
        template<Calculable TargetType, Calculable dType>
        auto sqrt(const Matrix<dType> &m) -> Matrix<TargetType> {
            Matrix<TargetType> r((Matrix<real_max>(m)));
            for (auto &i: r)
                i = std::sqrt(i);
            return r;
        }

        template<RealType RT>
        auto sqrt(const Matrix<RT> &m) -> Matrix<complex_max> {
            Matrix<complex_max> r((Matrix<real_max>(m)));
            for (Calculable auto &i: r)
                i = std::sqrt(i);
            return r;
        }

        template<ComplexType CT>
        auto sqrt(const Matrix<CT> &m) -> Matrix<complex_max> {
            Matrix<complex_max> r(m);
            for (Calculable auto &i: r)
                i = std::sqrt(i);
            return r;
        }

        template<RealType RT>
        auto realsqrt(const Matrix<RT> &m) -> Matrix<real_max> {
            Matrix<real_max> r(m);
            for (const Calculable auto &i: r)
                if (i < 0) throw_value_error("realsqrt运算数中不能包含负数");
            for (Calculable auto &i: r)
                i = std::sqrt(i);
            return r;
        }

        /**
         * @warning do not use
         */
        template<ComplexType CT>
        auto realsqrt(const Matrix<CT> &m) -> Matrix<complex_max> {
            throw_value_error("realsqrt不支持复数矩阵");
            return {};
        }

        template<Calculable lT, Calculable rT>
        auto mpower(const Matrix<lT> &left, const Matrix<rT> &right) -> decltype(left ^ right) {
            return left ^ right;
        }

        template<Calculable lT, Calculable rT>
        auto mpower(const Matrix<lT> &left, const rT &right) -> decltype(left ^ right) {
            return left ^ right;
        }

        template<Calculable lT, Calculable rT>
        auto mpower(const lT &left, const Matrix<rT> &right) -> decltype(left ^ right) {
            return left ^ right;
        }

        template<RealType lT, IntType rT>
        auto realpow(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<real_max> {
            return left ^ right;
        }

        template<RealType lT, IntType rT>
        auto realpow(const Matrix<lT> &left, const rT &right) -> Matrix<real_max> {
            return left ^ right;
        }

        template<RealType lT, IntType rT>
        auto realpow(const lT &left, const Matrix<rT> &right) -> Matrix<real_max> {
            return left ^ right;
        }

        template<RealType lT, FloatType rT>
        auto realpow(const Matrix<lT> &left, const Matrix<rT> &right) -> Matrix<real_max> {
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

        template<RealType lT, FloatType rT>
        auto realpow(const Matrix<lT> &left, const rT &right) -> Matrix<real_max> {
            Matrix<real_max> tmp(left);
            MatrixBase::Pow(tmp.getData(), right, tmp.getSize(), tmp.getData());
            return tmp;
        }

        template<RealType lT, FloatType rT>
        auto realpow(const lT &left, const Matrix<rT> &right) -> Matrix<real_max> {
            Matrix<real_max> tmp(right);
            MatrixBase::Pow(left, tmp.getData(), tmp.getSize(), tmp.getData());
            return tmp;
        }
    }

    ///matrix alter
    namespace Math {
        template<Calculable dType>
        auto T(const Matrix<dType> &m) -> Matrix<dType> {
            return m.T();
        }

        template<RealType RT>
        auto abs(const Matrix<RT> &m) -> Matrix<RT> {
            Matrix<RT> r(m);
            for (auto &i: r)
                i = std::abs(i);
            return r;
        }

        template<ComplexType CT>
        auto abs(const Matrix<CT> &m) -> Matrix<float_max> {
            Matrix<float_max> r(m.getShape(), 0.);
            for (Calculable auto &i: r)
                i = std::abs(i);
            return r;
        }

//        template<Calculable dType>
//        auto det(const Matrix<dType> &m) -> decltype(m.det()) {
//            return m.det();
//        }

//        template<Calculable dType>
//        auto inv(const Matrix<dType> &m) -> decltype(m.inv()) {
//            return m.inv();
//        }

        template<Calculable dType>
        auto cat(const vector<Matrix<dType>> &mv, int64 dim = 1) -> Matrix<dType> {
            return {mv, dim};
        }

        template<Calculable dType>
        auto cat(const initializer_list<Matrix<dType>> &mv, int64 dim = 1) -> Matrix<dType> {
            return {mv, dim};
        }
    }

    ///check matrix (iss)
    namespace Math {
        template<Calculable dType>
        auto isSquare(const Matrix<dType> &m) -> bool {
            return m.isSquare();
        }

        template<Calculable dType>
        auto isDiag(const Matrix<dType> &m) -> bool {
            return m.isDiag();
        }

        template<Calculable dType>
        auto isEmpty(const Matrix<dType> &m) -> bool {
            return m.isEmpty();
        }
    }
}
#endif //MATSAR_MATRIXOPERATION_HPP
