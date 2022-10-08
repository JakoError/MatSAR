//
// Created by JakoError on 2022/7/10.
//

#ifndef MATSAR_NUMBEROPERATION_HPP
#define MATSAR_NUMBEROPERATION_HPP

#include <string>
#include <bitset>
#include <numeric>

using namespace MatSAR;
using namespace std;

namespace MatSAR {
    static auto base2dec(const string &str, int base) -> int64 {
        if (base <= 0)
            throw_value_error("进制转换不能为负或0 输入base = " + to_string(base));
        const char *c_str = str.c_str();
        char *p;
        auto num = strtoll(c_str, &p, base);
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
        vector<int64> nums(strs.size(), 0);
        for (size_t i = 0; i < strs.size(); ++i)
            nums[i] = base2dec(data(strs)[i], base);
        return nums;
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

    template<IntType IT>
    static auto dec2base(IT v, int base) -> string {
        if (base <= 0 || base > 32)
            throw_value_error("进制转换不能为负或0 最大进制为36 输入base = " + to_string(base));
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
                str.insert(str.begin(), reVal(v % base));
                v /= base;
            }
            if (sign)
                str.insert(str.begin(), '-');
            ss << str;
        }
        return ss.str();
    }

    template<IntType IT>
    static auto dec2base(const vector<IT> &nums, int base) -> vector<string> {
        vector<string> strs(nums.size());
        for (size_t i = 0; i < strs.size(); ++i)
            strs[i] = dec2base(nums[i], base);
        return strs;
    }

    template<IntType IT>
    static auto dec2base(const initializer_list<IT> &nums, int base) -> vector<string> {
        vector<string> strs(nums.size());
        for (size_t i = 0; i < strs.size(); ++i)
            strs[i] = dec2base(data(nums)[i], base);
        return strs;
    }

    template<IntType IT>
    static auto dec2bin(IT v) -> string {
        return dec2base(v, 2);
    }

    template<IntType IT>
    static auto dec2bin(const vector<IT> &nums, int base) -> vector<string> {
        return dec2base(nums, 2);
    }

    template<IntType IT>
    static auto dec2bin(const initializer_list<IT> &nums, int base) -> vector<string> {
        return dec2base(nums, 2);
    }

    template<IntType IT>
    static auto dec2oct(IT v) -> string {
        return dec2base(v, 8);
    }

    template<IntType IT>
    static auto dec2oct(const vector<IT> &nums) -> vector<string> {
        return dec2base(nums, 8);
    }

    template<IntType IT>
    static auto dec2oct(const initializer_list<IT> &nums) -> vector<string> {
        return dec2base(nums, 8);
    }

    template<IntType IT>
    static auto dec2hex(IT v) -> string {
        return dec2base(v, 16);
    }

    template<IntType IT>
    static auto dec2hex(const vector<IT> &nums) -> vector<string> {
        return dec2base(nums, 16);
    }

    template<IntType IT>
    static auto dec2hex(const initializer_list<IT> &nums) -> vector<string> {
        return dec2base(nums, 16);
    }

    static auto hex2num(const string &str) -> float64 {
        int64 number = hex2dec(str);
        return *(float64 *) &number;
    }

    static auto hex2num(const vector<string> &strs) -> vector<float64> {
        vector<float64> nums(strs.size());
        for (size_t i = 0; i < strs.size(); ++i)
            nums[i] = hex2num(strs[i]);
        return nums;
    }

    static auto hex2num(const initializer_list<string> &strs) -> vector<float64> {
        vector<float64> nums(strs.size());
        for (size_t i = 0; i < strs.size(); ++i)
            nums[i] = hex2num(data(strs)[i]);
        return nums;
    }

    template<ComplexType CT>
    static auto ceil(const CT &v) -> CT {
        return CT(std::ceil(v.real()), std::ceil(v.imag()));
    }

    template<RealType RT>
    static auto ceil(const RT &v) -> RT {
        return std::ceil(v);
    }

    template<ComplexType CT>
    static auto floor(const CT &v) -> CT {
        return CT(std::floor(v.real()), std::floor(v.imag()));
    }

    template<RealType RT>
    static auto floor(const RT &v) -> RT {
        return std::floor(v);
    }

    template<ComplexType CT>
    static auto round(const CT &v) -> CT {
        return CT(std::round(v.real()), std::round(v.imag()));
    }

    template<RealType RT>
    static auto round(const RT &v) -> RT {
        return std::round(v);
    }

    template<Calculable T>
    static auto fix(const T &v) -> T {
        return v < 0 ? MatSAR::ceil(v) : MatSAR::floor(v);
    }

    template<IntType IT>
    static auto isPrime(IT number) -> bool {
        if (number < 2) return false;
        if (number == 2) return true;
        if (number % 2 == 0) return false;
        for (int64 i = 3; (i * i) <= number; i += 2) {
            if (number % i == 0) return false;
        }
        return true;
    }

    template<IntType T1, IntType T2>
    static auto numden(T1 n, T2 d) -> tuple<int_max, int_max> {
        bool sign = (n < 0 != d < 0);
        n = std::abs(n);
        d = std::abs(d);
        int_max div = std::gcd(n, d);
        if (sign)
            return {-n / div, d / div};
        else
            return {n / div, d / div};
    }

    template<RealType RT>
    inline static auto gamma(const RT &v) -> float_max {
        return std::tgamma(v);
    }

    template<RealType RT>
    inline static auto gammaln(const RT &v) -> float_max {
        if (v < 0) throw_value_error("gammaln: 要求传入值为非负数");
        return std::lgamma(v);
    }
}

/**
 * 虚数比较为取模(absolute value)比较
 */
template<ComplexType CT1, ComplexType CT2>
inline static auto operator<(const CT1 &left, const CT2 &right) -> bool {
    return std::abs(left) < std::abs(right);
}

/**
 * 虚数比较为取模(absolute value)比较
 */
template<ComplexType CT1, ComplexType CT2>
inline static auto operator<=(const CT1 &left, const CT2 &right) -> bool {
    return std::abs(left) <= std::abs(right);
}

/**
 * 虚数比较为取模(absolute value)比较
 */
template<ComplexType CT1, ComplexType CT2>
inline static auto operator>(const CT1 &left, const CT2 &right) -> bool {
    return std::abs(left) > std::abs(right);
}

/**
 * 虚数比较为取模(absolute value)比较
 */
template<ComplexType CT1, ComplexType CT2>
inline static auto operator>=(const CT1 &left, const CT2 &right) -> bool {
    return std::abs(left) >= std::abs(right);
}


/**
 * 虚数比较为取模(absolute value)比较
 */
template<RealType T1, ComplexType CT2>
inline static auto operator<(const T1 &left, const CT2 &right) -> bool {
    return left < std::abs(right);
}

/**
 * 虚数比较为取模(absolute value)比较
 */
template<RealType T1, ComplexType CT2>
inline static auto operator<=(const T1 &left, const CT2 &right) -> bool {
    return left <= std::abs(right);
}

/**
 * 虚数比较为取模(absolute value)比较
 */
template<RealType T1, ComplexType CT2>
inline static auto operator>(const T1 &left, const CT2 &right) -> bool {
    return left > std::abs(right);
}

/**
 * 虚数比较为取模(absolute value)比较
 */
template<RealType T1, ComplexType CT2>
inline static auto operator>=(const T1 &left, const CT2 &right) -> bool {
    return left >= std::abs(right);
}

/**
 * 虚数比较为取模(absolute value)比较
 */
template<ComplexType CT1, RealType T2>
inline static auto operator<(const CT1 &left, const T2 &right) -> bool {
    return std::abs(left) < right;
}

/**
 * 虚数比较为取模(absolute value)比较
 */
template<ComplexType CT1, RealType T2>
inline static auto operator<=(const CT1 &left, const T2 &right) -> bool {
    return std::abs(left) <= right;
}

/**
 * 虚数比较为取模(absolute value)比较
 */
template<ComplexType CT1, RealType T2>
inline static auto operator>(const CT1 &left, const T2 &right) -> bool {
    return std::abs(left) > right;
}

/**
 * 虚数比较为取模(absolute value)比较
 */
template<ComplexType CT1, RealType T2>
inline static auto operator>=(const CT1 &left, const T2 &right) -> bool {
    return std::abs(left) >= right;
}


///extra base function对基础函数的补充
template<ComplexType CT>
static auto real(const CT &v) -> float_max {
    return (float_max) v.real();
}

template<ComplexType CT>
static auto image(const CT &v) -> float_max {
    return (float_max) v.imag();
}

template<RealType RT>
static auto real(const RT &v) -> float_max {
    return (float_max) v;
}

template<RealType RT>
static auto image([[maybe_unused]]const RT &v) -> float_max {
    return 0.;
}

//template<RealType RT>
//static complex128 conj(const RT &v) {
//    return {(float_max) v, 0};
//}

template<ComplexType CT>
static auto log2(const CT &v) -> complex128 {
    return log(v) * static_cast<float_max>(1.442695040888963387004650940070860087871551513671875);

}

template<ComplexType CT>
static auto log10(const CT &v) -> complex128 {
    return log(v) * static_cast<float_max>(0.434294481903251761156781185491126962006092071533203125);
}

template<SIntType T1, SIntType T2>
static auto rem(const T1 &left, const T2 &right) -> real_max {
    if (right == 0) return NAN;
    int_max left_ = left;
    int_max right_ = right;
    int_max r = left_ - (left_ / right_) * right_;
    return real_max(r);
}

template<UIntType T1, UIntType T2>
static auto rem(const T1 &left, const T2 &right) -> real_max {
    if (right == 0) return NAN;
    int_max left_ = left;
    int_max right_ = right;
    int_max r = left_ - (left_ / right_) * right_;
    return real_max(r);
}

template<FloatType T1, FloatType T2>
static auto rem(const T1 &left, const T2 &right) -> float_max {
    if (right == 0.) return NAN;
    float_max left_ = left;
    float_max right_ = right;
    return left_ - fix(left_ / right_) * right_;
}

template<SIntType T1, SIntType T2>
static auto mod(const T1 &left, const T2 &right) -> int_max {
    if (right == 0) return NAN;
    int_max left_ = left;
    int_max right_ = right;
    int_max r = left_ - (left_ / right_) * right_;
    return right_ < 0 == r < 0 ? r : r + right_;
}

template<UIntType T1, UIntType T2>
static auto mod(const T1 &left, const T2 &right) -> uint_max {
    if (right == 0) return NAN;
    uint_max left_ = left;
    uint_max right_ = right;
    return left_ - (left_ / right_) * right_;
}

template<FloatType T1, FloatType T2>
static auto mod(const T1 &left, const T2 &right) -> float_max {
    if (right == 0.) return NAN;
    float_max left_ = left;
    float_max right_ = right;
    float_max r = left_ - floor(left_ / right_) * right_;
    return left_ < 0 == r < 0 ? r : r + right_;
}

template<Calculable T, class targetType>
static auto sqrt(const T &v) -> targetType {
    return std::sqrt(targetType(v));
}


#endif //MATSAR_NUMBEROPERATION_HPP
