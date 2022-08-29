//
// Created by JakoError on 2022/7/9.
//

#ifndef MATSAR_TRIGONOMETRY_HPP
#define MATSAR_TRIGONOMETRY_HPP

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

namespace MatSAR::Math {

    template<RealType RT>
    auto sin(const RT &v) -> real_max {
        return std::sin(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto sin(const CT &v) -> complex_max {
        return std::sin(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto sinh(const RT &v) -> real_max {
        return std::sinh(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto sinh(const CT &v) -> complex_max {
        return std::sinh(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto asin(const RT &v) -> real_max {
        return std::asin(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto asin(const CT &v) -> complex_max {
        return std::asin(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto asinh(const RT &v) -> real_max {
        return std::asinh(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto asinh(const CT &v) -> complex_max {
        return std::asinh(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto sin(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = sin(i);
        return r;
    }

    template<ComplexType CT>
    auto sin(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = sin(i);
        return r;
    }

    template<RealType RT>
    auto sinh(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = sinh(i);
        return r;
    }

    template<ComplexType CT>
    auto sinh(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = sinh(i);
        return r;
    }

    template<RealType RT>
    auto asin(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = asin(i);
        return r;
    }

    template<ComplexType CT>
    auto asin(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = asin(i);
        return r;
    }

    template<RealType RT>
    auto asinh(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = asinh(i);
        return r;
    }

    template<ComplexType CT>
    auto asinh(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = asinh(i);
        return r;
    }

    template<RealType RT>
    auto cos(const RT &v) -> real_max {
        return std::cos(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto cos(const CT &v) -> complex_max {
        return std::cos(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto cosh(const RT &v) -> real_max {
        return std::cosh(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto cosh(const CT &v) -> complex_max {
        return std::cosh(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto acos(const RT &v) -> real_max {
        return std::acos(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto acos(const CT &v) -> complex_max {
        return std::acos(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto acosh(const RT &v) -> real_max {
        return std::acosh(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto acosh(const CT &v) -> complex_max {
        return std::acosh(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto cos(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = cos(i);
        return r;
    }

    template<ComplexType CT>
    auto cos(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = cos(i);
        return r;
    }

    template<RealType RT>
    auto cosh(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = cosh(i);
        return r;
    }

    template<ComplexType CT>
    auto cosh(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = cosh(i);
        return r;
    }

    template<RealType RT>
    auto acos(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = acos(i);
        return r;
    }

    template<ComplexType CT>
    auto acos(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = acos(i);
        return r;
    }

    template<RealType RT>
    auto acosh(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = acosh(i);
        return r;
    }

    template<ComplexType CT>
    auto acosh(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = acosh(i);
        return r;
    }

    template<RealType RT>
    auto tan(const RT &v) -> real_max {
        return std::tan(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto tan(const CT &v) -> complex_max {
        return std::tan(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto tanh(const RT &v) -> real_max {
        return std::tanh(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto tanh(const CT &v) -> complex_max {
        return std::tanh(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto atan(const RT &v) -> real_max {
        return std::atan(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto atan(const CT &v) -> complex_max {
        return std::atan(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto atanh(const RT &v) -> real_max {
        return std::atanh(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto atanh(const CT &v) -> complex_max {
        return std::atanh(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto tan(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = tan(i);
        return r;
    }

    template<ComplexType CT>
    auto tan(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = tan(i);
        return r;
    }

    template<RealType RT>
    auto tanh(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = tanh(i);
        return r;
    }

    template<ComplexType CT>
    auto tanh(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = tanh(i);
        return r;
    }

    template<RealType RT>
    auto atan(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = atan(i);
        return r;
    }

    template<ComplexType CT>
    auto atan(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = atan(i);
        return r;
    }

    template<RealType RT>
    auto atanh(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = atanh(i);
        return r;
    }

    template<ComplexType CT>
    auto atanh(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = atanh(i);
        return r;
    }

    template<RealType RT>
    auto cot(const RT &v) -> real_max {
        return 1. / std::tan(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto cot(const CT &v) -> complex_max {
        return 1. / std::tan(static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto coth(const RT &v) -> real_max {
        return 1. / std::tanh(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto coth(const CT &v) -> complex_max {
        return 1. / std::tanh(static_cast<complex_max>(v));
    }

    ///atan2
    template<RealType RT>
    auto acot(const RT &v) -> real_max {
        return std::atan2(1., static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto acot(const CT &v) -> complex_max {
        return std::atan(1. / static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto acoth(const RT &v) -> real_max {
        return std::atanh(1. / static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto acoth(const CT &v) -> complex_max {
        return std::atanh(1. / static_cast<complex_max>(v));
    }

    template<RealType RT>
    auto cot(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = cot(i);
        return r;
    }

    template<ComplexType CT>
    auto cot(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = cot(i);
        return r;
    }

    template<RealType RT>
    auto coth(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = coth(i);
        return r;
    }

    template<ComplexType CT>
    auto coth(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = coth(i);
        return r;
    }

    template<RealType RT>
    auto acot(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = acot(i);
        return r;
    }

    template<ComplexType CT>
    auto acot(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = acot(i);
        return r;
    }

    template<RealType RT>
    auto acoth(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = acoth(i);
        return r;
    }

    template<ComplexType CT>
    auto acoth(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = acoth(i);
        return r;
    }

    template<RealType RT>
    auto sec(const RT &v) -> real_max {
        return 1. / std::cos(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto sec(const CT &v) -> complex_max {
        return 1. / std::cos(static_cast<real_max>(v));
    }

    template<RealType RT>
    auto sech(const RT &v) -> real_max {
        return 1. / std::cosh(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto sech(const CT &v) -> complex_max {
        return 1. / std::cosh(static_cast<real_max>(v));
    }

    template<RealType RT>
    auto asec(const RT &v) -> real_max {
        return std::acos(1. / static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto asec(const CT &v) -> complex_max {
        return std::acos(1. / static_cast<real_max>(v));
    }

    template<RealType RT>
    auto asech(const RT &v) -> real_max {
        return std::acosh(1. / static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto asech(const CT &v) -> complex_max {
        return std::acosh(1. / static_cast<real_max>(v));
    }

    template<RealType RT>
    auto sec(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = sec(i);
        return r;
    }

    template<ComplexType CT>
    auto sec(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = sec(i);
        return r;
    }

    template<RealType RT>
    auto sech(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = sech(i);
        return r;
    }

    template<ComplexType CT>
    auto sech(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = sech(i);
        return r;
    }

    template<RealType RT>
    auto asec(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = asec(i);
        return r;
    }

    template<ComplexType CT>
    auto asec(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = asec(i);
        return r;
    }

    template<RealType RT>
    auto asech(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = asech(i);
        return r;
    }

    template<ComplexType CT>
    auto asech(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = asech(i);
        return r;
    }

    template<RealType RT>
    auto csc(const RT &v) -> real_max {
        return 1. / std::sin(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto csc(const CT &v) -> complex_max {
        return 1. / std::sin(static_cast<real_max>(v));
    }

    template<RealType RT>
    auto csch(const RT &v) -> real_max {
        return 1. / std::sinh(static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto csch(const CT &v) -> complex_max {
        return 1. / std::sinh(static_cast<real_max>(v));
    }

    template<RealType RT>
    auto acsc(const RT &v) -> real_max {
        return std::asin(1. / static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto acsc(const CT &v) -> complex_max {
        return std::asin(1. / static_cast<real_max>(v));
    }

    template<RealType RT>
    auto acsch(const RT &v) -> real_max {
        return std::asinh(1. / static_cast<real_max>(v));
    }

    template<ComplexType CT>
    auto acsch(const CT &v) -> complex_max {
        return std::asinh(1. / static_cast<real_max>(v));
    }

    template<RealType RT>
    auto csc(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = csc(i);
        return r;
    }

    template<ComplexType CT>
    auto csc(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = csc(i);
        return r;
    }

    template<RealType RT>
    auto csch(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = csch(i);
        return r;
    }

    template<ComplexType CT>
    auto csch(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = csch(i);
        return r;
    }

    template<RealType RT>
    auto acsc(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = acsc(i);
        return r;
    }

    template<ComplexType CT>
    auto acsc(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = acsc(i);
        return r;
    }

    template<RealType RT>
    auto acsch(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = acsch(i);
        return r;
    }

    template<ComplexType CT>
    auto acsch(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = acsch(i);
        return r;
    }


    template<RealType RT>
    auto atan2(const RT &v) -> real_max {
        return std::atan2(static_cast<real_max>(v), 1.);
    }

    template<ComplexType CT>
    auto atan2(const CT &v) -> complex_max {
        return std::atan2(static_cast<real_max>(v), 1.);
    }

    template<RealType RT>
    auto atan2(const Matrix<RT> &m) -> Matrix<real_max> {
        Matrix<real_max> r(m);
        for (auto &i: r)
            i = atan2(i);
        return r;
    }

    template<ComplexType CT>
    auto atan2(const Matrix<CT> &m) -> Matrix<complex_max> {
        Matrix<complex_max> r(m);
        for (auto &i: r)
            i = atan2(i);
        return r;
    }

    template<RealType RT1, RealType RT2>
    auto atan2(const RT1 &y, const RT2 &x) -> real_max {
        return std::atan2(static_cast<real_max>(y), static_cast<real_max>(x));
    }

    template<RealType RT1, RealType RT2>
    auto atan2(const Matrix<RT1> &Y, const Matrix<RT1> &X) -> Matrix<real_max> {
        if (Y.getShape() == X.getShape()) {
            Matrix<real_max> r(Y.getShape(), 0.);
            auto iter_y = Y.getData();
            auto iter_x = X.getData();
            for (auto &i: r)
                i = atan2(*iter_y++, *iter_x++);
            return r;
        } else {
            Matrix<real_max> b_Y(Y);
            Matrix<real_max> b_X(X);
            Matrix<real_max>::broadcast(b_Y, b_X);
            Matrix<real_max> r(b_Y.getShape(), 0.);
            auto iter_y = b_Y.getData();
            auto iter_x = b_X.getData();
            for (auto &i: r)
                i = atan2(*iter_y++, *iter_x++);
            return r;
        }
    }
}

#endif //MATSAR_TRIGONOMETRY_HPP
