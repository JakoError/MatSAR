// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2022/7/9.
//

#ifndef MATSAR_TRIGONOMETRY_HPP
#define MATSAR_TRIGONOMETRY_HPP

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

// require universal matrix functions
#ifndef MATSAR_UNIVERSALFUNCTIONS_HPP

#include "../UniversalFunctions.hpp"

#endif //MATSAR_UNIVERSALFUNCTIONS_HPP

namespace matsar {
    namespace Math {
        template<typename T>
        auto Sin(const T &v) -> decltype(std::sin(v + 0.)) {
            return std::sin(static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Sinh(const T &v) -> decltype(std::sinh(v + 0.)) {
            return std::sinh(static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Asin(const T &v) -> decltype(std::asin(v + 0.)) {
            return std::asin(static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Asinh(const T &v) -> decltype(std::asinh(v + 0.)) {
            return std::asinh(static_cast<decltype(v + 0.)>(v));
        }

        template<typename dType>
        inline auto sin(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Sin<dType>)) {
            return matsar::apply_function(m, matsar::Math::Sin<dType>);
        }

        template<typename dType>
        inline auto sinh(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Sinh<dType>)) {
            return matsar::apply_function(m, matsar::Math::Sinh<dType>);
        }

        template<typename dType>
        inline auto asin(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Asin<dType>)) {
            return matsar::apply_function(m, matsar::Math::Asin<dType>);
        }

        template<typename dType>
        inline auto asinh(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Asinh<dType>)) {
            return matsar::apply_function(m, matsar::Math::Asinh<dType>);
        }
    }

    namespace Math {
        template<typename T>
        auto Cos(const T &v) -> decltype(std::cos(v + 0.)) {
            return std::cos(static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Cosh(const T &v) -> decltype(std::cosh(v + 0.)) {
            return std::cosh(static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Acos(const T &v) -> decltype(std::acos(v + 0.)) {
            return std::acos(static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Acosh(const T &v) -> decltype(std::acosh(v + 0.)) {
            return std::acosh(static_cast<decltype(v + 0.)>(v));
        }

        template<typename dType>
        inline auto cos(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Cos<dType>)) {
            return matsar::apply_function(m, matsar::Math::Cos<dType>);
        }

        template<typename dType>
        inline auto cosh(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Cosh<dType>)) {
            return matsar::apply_function(m, matsar::Math::Cosh<dType>);
        }

        template<typename dType>
        inline auto acos(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Acos<dType>)) {
            return matsar::apply_function(m, matsar::Math::Acos<dType>);
        }

        template<typename dType>
        inline auto acosh(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Acosh<dType>)) {
            return matsar::apply_function(m, matsar::Math::Acosh<dType>);
        }
    }

    namespace Math {
        template<typename T>
        auto Tan(const T &v) -> decltype(std::tan(v + 0.)) {
            return std::tan(static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Tanh(const T &v) -> decltype(std::tanh(v + 0.)) {
            return std::tanh(static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Atan(const T &v) -> decltype(std::atan(v + 0.)) {
            return std::atan(static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Atanh(const T &v) -> decltype(std::atanh(v + 0.)) {
            return std::atanh(static_cast<decltype(v + 0.)>(v));
        }

        template<typename dType>
        inline auto tan(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Tan<dType>)) {
            return matsar::apply_function(m, matsar::Math::Tan<dType>);
        }

        template<typename dType>
        inline auto tanh(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Tanh<dType>)) {
            return matsar::apply_function(m, matsar::Math::Tanh<dType>);
        }

        template<typename dType>
        inline auto atan(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Atan<dType>)) {
            return matsar::apply_function(m, matsar::Math::Atan<dType>);
        }

        template<typename dType>
        inline auto atanh(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Atanh<dType>)) {
            return matsar::apply_function(m, matsar::Math::Atanh<dType>);
        }
    }

    namespace Math {
        template<typename T>
        auto Cot(const T &v) -> decltype(1. / matsar::Math::Tan(v)) {
            return 1. / matsar::Math::Tan(v);
        }

        template<typename T>
        auto Coth(const T &v) -> decltype(1. / matsar::Math::Tanh(v)) {
            return 1. / matsar::Math::Tanh(v);
        }

//        template<typename T>
//        auto Acot(const complex<T> &v) -> decltype(std::atan(1./ static_cast<decltype(v + 0.)>(v))) {
//            return std::atan(1./ static_cast<decltype(v + 0.)>(v));
//        }

        template<typename T>
        auto Acot(const T &v) -> decltype(std::atan(1. / static_cast<decltype(v + 0.)>(v))) {
            return std::atan(1. / static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Acoth(const T &v) -> decltype(std::atanh(1. / static_cast<decltype(v + 0.)>(v))) {
            return std::atanh(1. / static_cast<decltype(v + 0.)>(v));
        }

        template<typename dType>
        inline auto cot(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Cot<dType>)) {
            return matsar::apply_function(m, matsar::Math::Cot<dType>);
        }

        template<typename dType>
        inline auto coth(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Coth<dType>)) {
            return matsar::apply_function(m, matsar::Math::Coth<dType>);
        }

        template<typename dType>
        inline auto acot(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Acot<dType>)) {
            return matsar::apply_function(m, matsar::Math::Acot<dType>);
        }

        template<typename dType>
        inline auto acoth(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Acoth<dType>)) {
            return matsar::apply_function(m, matsar::Math::Acoth<dType>);
        }
    }

    namespace Math {
        template<typename T>
        auto Sec(const T &v) -> decltype(1. / matsar::Math::Cos(v)) {
            return 1. / matsar::Math::Cos(v);
        }

        template<typename T>
        auto Sech(const T &v) -> decltype(1. / matsar::Math::Cosh(v)) {
            return 1. / matsar::Math::Cosh(v);
        }

        template<typename T>
        auto Asec(const T &v) -> decltype(std::acos(1. / static_cast<decltype(v + 0.)>(v))) {
            return std::acos(1. / static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Asech(const T &v) -> decltype(std::acosh(1. / static_cast<decltype(v + 0.)>(v))) {
            return std::acosh(1. / static_cast<decltype(v + 0.)>(v));
        }

        template<typename dType>
        inline auto sec(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Sec<dType>)) {
            return matsar::apply_function(m, matsar::Math::Sec<dType>);
        }

        template<typename dType>
        inline auto sech(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Sech<dType>)) {
            return matsar::apply_function(m, matsar::Math::Sech<dType>);
        }

        template<typename dType>
        inline auto asec(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Asec<dType>)) {
            return matsar::apply_function(m, matsar::Math::Asec<dType>);
        }

        template<typename dType>
        inline auto asech(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Asech<dType>)) {
            return matsar::apply_function(m, matsar::Math::Asech<dType>);
        }
    }

    namespace Math {
        template<typename T>
        auto Csc(const T &v) -> decltype(1. / matsar::Math::Sin(v)) {
            return 1. / matsar::Math::Sin(v);
        }

        template<typename T>
        auto Csch(const T &v) -> decltype(1. / matsar::Math::Sinh(v)) {
            return 1. / matsar::Math::Sinh(v);
        }

        template<typename T>
        auto Acsc(const T &v) -> decltype(std::asin(1. / static_cast<decltype(v + 0.)>(v))) {
            return std::asin(1. / static_cast<decltype(v + 0.)>(v));
        }

        template<typename T>
        auto Acsch(const T &v) -> decltype(std::asinh(1. / static_cast<decltype(v + 0.)>(v))) {
            return std::asinh(1. / static_cast<decltype(v + 0.)>(v));
        }

        template<typename dType>
        inline auto csc(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Csc<dType>)) {
            return matsar::apply_function(m, matsar::Math::Csc<dType>);
        }

        template<typename dType>
        inline auto csch(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Csch<dType>)) {
            return matsar::apply_function(m, matsar::Math::Csch<dType>);
        }

        template<typename dType>
        inline auto acsc(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Acsc<dType>)) {
            return matsar::apply_function(m, matsar::Math::Acsc<dType>);
        }

        template<typename dType>
        inline auto acsch(const Matrix<dType> &m) -> decltype(matsar::apply_function(m, matsar::Math::Acsch<dType>)) {
            return matsar::apply_function(m, matsar::Math::Acsch<dType>);
        }
    }
    namespace Math {
        template<typename T1, typename T2>
        auto Atan2(const T1 &y, const T2 &x) -> real_max {
            return std::atan2(static_cast<real_max>(y), static_cast<real_max>(x));
        }

        template<typename T1, typename T2>
        auto atan2(Matrix<T1> Y, Matrix<T2> X) -> Matrix<real_max> {
            Matrix<real_max>::broadcast(Y, X);
            return matsar::apply_function(Y, X, matsar::Math::Atan2<T1, T2>);
        }
    }
    namespace Math {
        template<typename dType>
        auto angle(const Matrix<dType> &A) -> Matrix<float64> {
            Matrix<float64> r(A.getShape(), 0.);
            auto iter_r = r.begin();
            for (auto iter_a = A.begin(); iter_a != A.end(); ++iter_a, ++iter_r) {
                auto A_r = real(*iter_a);
                auto A_i = imag(*iter_a);
                if (A_i == 0.) {
                    *iter_r = A_r < 0 ? pi : 0;
                } else {
                    *iter_r = Atan2(A_i, A_r);//Atan2(x,y)=Atan(x/y)
                }
            }
            return r;
        }
    }

    namespace Math {
        template<typename T1, typename T2>
        auto pol2cart(Matrix<T1> &theta, Matrix<T2> &rho) -> vector<Matrix<real_max>> {
            return {mul(rho, cos(theta)), mul(rho, sin(theta))};
        }
    }
}
#endif //MATSAR_TRIGONOMETRY_HPP
