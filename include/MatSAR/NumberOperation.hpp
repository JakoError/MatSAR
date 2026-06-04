// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2022/7/10.
//
#ifndef MATSAR_NUMBEROPERATION_HPP
#pragma once
#define MATSAR_NUMBEROPERATION_HPP

#ifndef MATSAR_UTILS

#include "util/Utils"

#endif

#include <string>
#include <bitset>
#include <numeric>
#include <list>
#include <cstring>

//using namespace matsar;
//using namespace std;
using namespace matsar::Util::TypeUtil;

namespace matsar {
    namespace Math {
#ifdef M_PI
        static const float_max pi = M_PI;
        static const float_max PI = M_PI;
#else
        static const float_max pi = 3.14159265358979323846;
        static const float_max PI = 3.14159265358979323846;
#endif
    }
}

namespace matsar {
    static auto base2dec(const string &str, int base) -> int64 {
        if (base <= 0 || base > 36)
            throw_value_error("base must be an even number between 1 and 36, got base = " + to_string(base));
        const char *c_str = str.c_str();
        char *p;
        auto num = strtol(c_str, &p, base);
        if (num == 0 && p == c_str)
            throw_value_error("unable to convert str to number:" + str);
        return num;
    }

    static auto base2dec(const vector<string> &strs, int base) -> vector<int64> {
        vector<int64> nums(strs.size(), 0);
        for (size_t i = 0; i < strs.size(); ++i)
            nums[i] = base2dec(strs[i], base);
        return nums;
    }

    static auto base2dec(const initializer_list<string> &strs, int base) -> vector<int64> {
        return base2dec(vector<string>(strs), base);
    }

    static auto bin2dec(const string &str) -> int64 {
        return base2dec(str, 2);
    }

    static auto bin2dec(const vector<string> &strs) -> vector<int64> {
        return base2dec(strs, 2);
    }

    static auto bin2dec(const initializer_list<string> &strs) -> vector<int64> {
        return base2dec(strs, 2);
    }

    static auto oct2dec(const string &str) -> int64 {
        return base2dec(str, 8);
    }

    static auto oct2dec(const vector<string> &strs) -> vector<int64> {
        return base2dec(strs, 8);
    }

    static auto oct2dec(const initializer_list<string> &strs) -> vector<int64> {
        return base2dec(strs, 8);
    }

    static auto hex2dec(const string &str) -> int64 {
        return base2dec(str, 16);
    }

    static auto hex2dec(const vector<string> &strs) -> vector<int64> {
        return base2dec(strs, 16);
    }

    static auto hex2dec(const initializer_list<string> &strs) -> vector<int64> {
        return base2dec(strs, 16);
    }

    auto reVal(int num) -> char {
        if (num >= 0 && num <= 9)
            return (char) (num + '0');
        else
            return (char) (num - 10 + 'A');
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2base(IT v, int base) -> string {
        if (base <= 0 || base > 36)
            throw_value_error("base must be an even number between 1 and 36, got base = " + to_string(base));
        stringstream ss;
        if (base == 2) {
            auto bit = bitset<64>(v).to_string();
            bit.erase(0, bit.find_first_not_of('0'));
            ss << bit;
        } else if (base == 8)
            ss << oct << v;
        else if (base == 16)
            ss << hex << v;
        else {
            bool sign = v < 0;
            if (sign) v = -v;
            string str;
            while (v > 0) {
                str.insert(str.begin(), reVal((int) v % base));
                v /= base;
            }
            if (sign)
                str.insert(str.begin(), '-');
            ss << str;
        }
        return ss.str();
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2base(const vector<IT> &nums, int base) -> vector<string> {
        vector<string> strs(nums.size());
        auto iter_nums = nums.begin();
        for (auto &str: strs) {
            str = dec2base(*iter_nums, base);
            iter_nums++;
        }
        return strs;
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2base(const initializer_list<IT> &nums, int base) -> vector<string> {
        vector<string> strs(nums.size());
        auto iter_nums = nums.begin();
        for (auto &str: strs) {
            str = dec2base(*iter_nums, base);
            iter_nums++;
        }
        return strs;
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2bin(IT v) -> string {
        return dec2base(v, 2);
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2bin(const vector<IT> &nums, int base) -> vector<string> {
        return dec2base(nums, 2);
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2bin(const initializer_list<IT> &nums, int base) -> vector<string> {
        return dec2base(nums, 2);
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2oct(IT v) -> string {
        return dec2base(v, 8);
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2oct(const vector<IT> &nums) -> vector<string> {
        return dec2base(nums, 8);
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2oct(const initializer_list<IT> &nums) -> vector<string> {
        return dec2base(nums, 8);
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2hex(IT v) -> string {
        return dec2base(v, 16);
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2hex(const vector<IT> &nums) -> vector<string> {
        return dec2base(nums, 16);
    }

    template<IntType IT CheckInt(IT)>
    static auto dec2hex(const initializer_list<IT> &nums) -> vector<string> {
        return dec2base(nums, 16);
    }

    //wwl
    static auto hex2num(const std::string &hexStr) -> double {

        std::string paddedStr = hexStr;
        if (hexStr.size() < 16) {
            paddedStr.insert(paddedStr.end(), 16 - hexStr.size(), '0');
            std::cout << paddedStr << std::endl;
        }

        uint64_t hexValue;
        std::stringstream ss;
        ss << std::hex << paddedStr;
        ss >> hexValue;

        double result;
        memcpy(&result, &hexValue, sizeof(result));
        return result;
    }

    //wwl
    static auto hex2num(const std::vector<std::string> &hexStr) -> std::vector<double> {
        std::vector<double> result(hexStr.size());

        for (size_t i = 0; i < hexStr.size(); i++) {
            std::string paddedStr = hexStr[i];
            if (paddedStr.size() < 16) {
                paddedStr.insert(paddedStr.end(), 16 - paddedStr.size(), '0');
            }

            uint64_t hexValue;
            std::stringstream ss;
            ss << std::hex << paddedStr;
            ss >> hexValue;

            double value;
            memcpy(&value, &hexValue, sizeof(value));
            result[i] = value;
        }

        return result;
    }

    static auto hex2num(const initializer_list<string> &strs) -> vector<float64> {
        return hex2num(vector<string>(strs));
    }

    inline static auto ceil(const float_max &v) -> float_max {
        return std::ceil(v);
    }

    inline static auto ceil(const complex64 &v) -> complex64 {
        return {std::ceil(v.real()), std::ceil(v.imag())};
    }

    inline static auto ceil(const complex128 &v) -> complex128 {
        return {std::ceil(v.real()), std::ceil(v.imag())};
    }

    inline static auto floor(const float_max &v) -> float_max {
        return std::floor(v);
    }

    inline static auto floor(const complex64 &v) -> complex64 {
        return {std::floor(v.real()), std::floor(v.imag())};
    }

    inline static auto floor(const complex128 &v) -> complex128 {
        return {std::floor(v.real()), std::floor(v.imag())};
    }

    inline static auto round(const float_max &v) -> float_max {
        return std::round(v);
    }

    inline static auto round(const complex64 &v) -> complex64 {
        return {std::round(v.real()), std::round(v.imag())};
    }

    static auto round(const complex128 &v) -> complex128 {
        return {std::round(v.real()), std::round(v.imag())};
    }

    template<typename T>
    static auto fix(const T &v) -> T {
        return v < 0 ? matsar::ceil(v) : matsar::floor(v);
    }

    template<IntType IT CheckInt(IT)>
    static auto isPrime(const IT &number) -> bool {
        if (number < 2) return false;
        if (number == 2) return true;
        if (number % 2 == 0) return false;
        for (int64 i = 3; (i * i) <= number; i += 2) {
            if (number % i == 0) return false;
        }
        return true;
    }

    template<typename T>
    auto gcd(T u, T v) -> T {
        while (v != 0) {
            T r = u % v;
            u = v;
            v = r;
        }
        return u;
    }

    template<typename T1, typename T2, typename TargetType = decltype(std::declval<T1>() + std::declval<T2>() + std::declval<int64>())>
    static auto numden(T1 n, T2 d) -> vector<TargetType> {
        static_assert(Util::TypeUtil::isInt<T1>() && Util::TypeUtil::isInt<T1>(), "require integer");
        bool sign = (n < 0 != d < 0);
        n = std::abs(n);
        d = std::abs(d);
        auto div = gcd(n, d);
        if (sign) {
            return {-n / div, d / div};
        } else {
            return {n / div, d / div};
        }
    }

    /**
     *
     * @tparam T
     * @tparam TargetType The default value is float_max. In certain cases, factorial will overflow the int range.
     * @param i
     * @return
     */
    template<typename T, typename TargetType = float_max>
    auto factorial(T i) -> TargetType {
        if (i < 0)
            throw_value_error("Factorial cannot be negative");
        if (i != std::round(i))
            throw ValueError("Factorial requires an integer");
        TargetType r = 1;
        while (i > 1)
            r *= i--;
        return r;
    }

    template<typename RT1, typename RT2, typename TargetType = float_max>
    auto nchoosek(RT1 n, RT2 k) -> TargetType {
        if (n != matsar::round(n) || k != matsar::round(k))
            return NAN;
        if (n < 0 || k < 0) {
            return NAN;
        }
        if (k > n) {
            return NAN;
        }
        TargetType r = 1;
        for (RT2 i = 0; i < k; ++i)
            r *= n - i;
        for (RT2 j = 2; j <= k; ++j)
            r /= j;
        return r;
    }

    template<typename T>
    inline static auto gamma(const T &v) -> decltype(std::tgamma(v)) {
        return std::tgamma(v);
    }

    template<typename T>
    inline static auto gammaln(const T &v) -> decltype(std::lgamma(v)) {
        return std::lgamma(v);
    }

    template<typename T>
    inline static auto real(const T &v) -> decltype(std::real(v)) {
        return std::real(v);
    }

    template<typename T>
    inline static auto image(const T &v) -> decltype(std::imag(v)) {
        return std::imag(v);
    }

    template<typename T>
    inline static auto imag(const T &v) -> decltype(std::imag(v)) {
        return std::imag(v);
    }

    //use std:: instead
    template<typename T>
    inline static auto conj(const T &v) -> decltype(std::conj(v)) {
        return std::conj(v);
    }

    template<typename T>
    inline static auto log(const T &v) -> decltype(std::log(v)) {
        return std::log(v);
    }

    template<typename T>
    inline static auto log2(const T &v) -> decltype(std::log2(v)) {
        return std::log2(v);
    }

    template<typename T>
    static auto log2(const complex<T> &v) -> complex<T> {
        return std::log(v) * complex<T>(1.442695040888963387004650940070860087871551513671875);//1/ln(2)
    }

    template<typename T>
    inline static auto log10(const T &v) -> decltype(std::log10(v)) {
        return std::log10(v);
    }

    template<typename T>
    static auto log10(const complex<T> &v) -> complex<T> {
        return std::log(v) * complex<T>(0.434294481903251761156781185491126962006092071533203125);//1/ln(10)
    }

    template<typename T>
    inline static auto exp(const T &v) -> decltype(std::exp(v)) {
        return std::exp(v);
    }

    template<typename T>
    inline static auto exp2(const T &v) -> decltype(std::exp2(v)) {
        return std::exp2(v);
    }

    template<typename T1, typename T2>
    static auto rem(const T1 &left, const T2 &right) -> float_max {
        if (right == 0.) return NAN;
        auto left_ = static_cast<float_max>(left);
        auto right_ = static_cast<float_max>(right);
        return left_ - matsar::fix(left_ / right_) * right_;
    }

    template<typename T1, typename T2>
    static auto mod(const T1 &left, const T2 &right) -> float_max {
        if (right == 0.) return NAN;
        auto left_ = static_cast<float_max>(left);
        auto right_ = static_cast<float_max>(right);
        auto r = left_ - floor(left_ / right_) * right_;
        return left_ < 0 == r < 0 ? r : r + right_;
    }

    template<typename T>
    static auto sqrt(const T &v) -> decltype(std::sqrt(v)) {
        return std::sqrt(v);
    }

    template<typename T>
    static auto abs(const T &v) -> decltype(std::abs(v)) {
        return std::abs(v);
    }
}

namespace matsar {
//    template<Calculable T CheckCalculable(T)>
//    inline static constexpr auto operator+(const T &left, const bool &right) -> T {
//        return left + static_cast<T>(right ? 1 : 0);
//    }
//
//    template<Calculable T CheckCalculable(T)>
//    inline static constexpr auto operator+(const bool &left, const T &right) -> T {
//        return static_cast<T>(left ? 1 : 0) + right;
//    }
//
//    template<Calculable T CheckCalculable(T)>
//    inline static constexpr auto operator+=(T &left, const bool &right) -> T & {
//        return left += static_cast<T>(right ? 1 : 0);
//    }
//
//    template<Calculable T CheckCalculable(T)>
//    inline static constexpr auto operator-(const T &left, const bool &right) -> T {
//        return left - static_cast<T>(right ? 0 : 1);
//    }
//
//    template<Calculable T CheckCalculable(T)>
//    inline static constexpr auto operator-(const bool &left, const T &right) -> T {
//        return static_cast<T>(left ? -1 : 1) - right;
//    }
//
//    template<Calculable T CheckCalculable(T)>
//    inline static constexpr auto operator-=(T &left, const bool &right) -> T & {
//        return left -= static_cast<T>(right ? -1 : 1);
//    }

    /**
     * Imaginary comparison is modulo (absolute value) comparison
     */
    template<typename T1, typename T2>
    inline static constexpr auto operator<(const complex<T1> &left, const complex<T2> &right) -> bool {
        return std::abs(left) < std::abs(right);
    }

    /**
     * Imaginary comparison is modulo (absolute value) comparison
     */
    template<typename T1, typename T2>
    inline static constexpr auto operator<=(const complex<T1> &left, const complex<T2> &right) -> bool {
        return std::abs(left) <= std::abs(right);
    }

    /**
     * Imaginary comparison is modulo (absolute value) comparison
     */
    template<typename T1, typename T2>
    inline static constexpr auto operator>(const complex<T1> &left, const complex<T2> &right) -> bool {
        return std::abs(left) > std::abs(right);
    }

    /**
     * Imaginary comparison is modulo (absolute value) comparison
     */
    template<typename T1, typename T2>
    inline static constexpr auto operator>=(const complex<T1> &left, const complex<T2> &right) -> bool {
        return std::abs(left) >= std::abs(right);
    }

///**
// * Imaginary comparison is modulo (absolute value) comparison
// */
//template<typename T1, typename T2>
//inline static auto operator<(const T1 &left, const complexT2 &right) -> bool {
//    return left < std::abs(right);
//}
//
///**
// * Imaginary comparison is modulo (absolute value) comparison
// */
//template<RealType T1, ComplexType CT2>
//inline static auto operator<=(const T1 &left, const CT2 &right) -> bool {
//    return left <= std::abs(right);
//}
//
///**
// * Imaginary comparison is modulo (absolute value) comparison
// */
//template<RealType T1, ComplexType CT2>
//inline static auto operator>(const T1 &left, const CT2 &right) -> bool {
//    return left > std::abs(right);
//}
//
///**
// * Imaginary comparison is modulo (absolute value) comparison
// */
//template<RealType T1, ComplexType CT2>
//inline static auto operator>=(const T1 &left, const CT2 &right) -> bool {
//    return left >= std::abs(right);
//}
//
///**
// * Imaginary comparison is modulo (absolute value) comparison
// */
//template<ComplexType CT1, RealType T2>
//inline static auto operator<(const CT1 &left, const T2 &right) -> bool {
//    return std::abs(left) < right;
//}
//
///**
// * Imaginary comparison is modulo (absolute value) comparison
// */
//template<ComplexType CT1, RealType T2>
//inline static auto operator<=(const CT1 &left, const T2 &right) -> bool {
//    return std::abs(left) <= right;
//}
//
///**
// * Imaginary comparison is modulo (absolute value) comparison
// */
//template<ComplexType CT1, RealType T2>
//inline static auto operator>(const CT1 &left, const T2 &right) -> bool {
//    return std::abs(left) > right;
//}
//
///**
// * Imaginary comparison is modulo (absolute value) comparison
// */
//template<ComplexType CT1, RealType T2>
//inline static auto operator>=(const CT1 &left, const T2 &right) -> bool {
//    return std::abs(left) >= right;
//}
}
#endif //MATSAR_NUMBEROPERATION_HPP
