//
// Created by JakoError on 2022/7/2.
//
#pragma once
#ifndef MATSAR_TYPEUTIL_HPP
#define MATSAR_TYPEUTIL_HPP

#include <cmath>
#include <complex>
#include <string>

#include "UtilHeadCheck.hpp"

using std::complex;
using std::string;

//int all type
typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

//float all type
typedef float float32;
typedef double float64;

//complex all type
typedef complex<float32> complex64;
typedef complex<float64> complex128;
//require float128
//typedef complex<float128> complex256;

typedef int64 int_max;
typedef uint64 uint_max;
typedef complex128 complex_max;
typedef float64 float_max;
typedef float_max real_max;

template<typename dType>
concept Calculable = std::is_same<dType, int8>::value ||
                     std::is_same<dType, int16>::value ||
                     std::is_same<dType, int32>::value ||
                     std::is_same<dType, int64>::value ||
                     std::is_same<dType, uint8>::value ||
                     std::is_same<dType, uint16>::value ||
                     std::is_same<dType, uint32>::value ||
                     std::is_same<dType, uint64>::value ||
                     std::is_same<dType, float32>::value ||
                     std::is_same<dType, float64>::value ||
                     std::is_same<dType, complex64>::value ||
                     std::is_same<dType, complex128>::value;


template<typename dType>
concept IntType = std::is_same<dType, int8>::value ||
                  std::is_same<dType, int16>::value ||
                  std::is_same<dType, int32>::value ||
                  std::is_same<dType, int64>::value ||
                  std::is_same<dType, uint8>::value ||
                  std::is_same<dType, uint16>::value ||
                  std::is_same<dType, uint32>::value ||
                  std::is_same<dType, uint64>::value;

template<typename dType>
concept SIntType = std::is_same<dType, int8>::value ||
                   std::is_same<dType, int16>::value ||
                   std::is_same<dType, int32>::value ||
                   std::is_same<dType, int64>::value;

template<typename dType>
concept UIntType = std::is_same<dType, uint8>::value ||
                   std::is_same<dType, uint16>::value ||
                   std::is_same<dType, uint32>::value ||
                   std::is_same<dType, uint64>::value;

template<typename dType>
concept FloatType = std::is_same<dType, float32>::value ||
                    std::is_same<dType, float64>::value;

template<typename dType>
concept RealType = std::is_same<dType, int8>::value ||
                   std::is_same<dType, int16>::value ||
                   std::is_same<dType, int32>::value ||
                   std::is_same<dType, int64>::value ||
                   std::is_same<dType, uint8>::value ||
                   std::is_same<dType, uint16>::value ||
                   std::is_same<dType, uint32>::value ||
                   std::is_same<dType, uint64>::value ||
                   std::is_same<dType, float32>::value ||
                   std::is_same<dType, float64>::value;

template<typename dType>
concept ComplexType = std::is_same<dType, complex64>::value ||
                      std::is_same<dType, complex128>::value;

enum CalculableLevel {
    int8_level, uint8_level,
    int16_level, uint16_level,
    int32_level, uint32_level,
    int64_level, uint64_level,
    float32_level, float64_level,
    complex64_level, complex128_level
};

#define LIST1 initializer_list<dType>
#define LIST2 initializer_list<LIST1>
#define LIST3 initializer_list<LIST2>
#define LIST4 initializer_list<LIST3>
#define LIST5 initializer_list<LIST4>
#define LIST6 initializer_list<LIST5>
#define LIST7 initializer_list<LIST6>
#define LIST8 initializer_list<LIST7>

#define VEC1 vector<dType>
#define VEC2 vector<VEC1>
#define VEC3 vector<VEC2>
#define VEC4 vector<VEC3>
#define VEC5 vector<VEC4>
#define VEC6 vector<VEC5>
#define VEC7 vector<VEC6>
#define VEC8 vector<VEC7>

namespace MatSAR::Util::TypeUtil {
    template<Calculable dType>
    static constexpr string getCalculableString() {
        ///COMMON_USE
        if (std::is_same<dType, int32>::value)
            return "int32";
        if (std::is_same<dType, int64>::value)
            return "int64";
        if (std::is_same<dType, float32>::value)
            return "float32";
        if (std::is_same<dType, float64>::value)
            return "float64";
        if (std::is_same<dType, complex64>::value)
            return "complex64";
        if (std::is_same<dType, complex128>::value)
            return "complex128";

        ///RARE_USE
        if (std::is_same<dType, int8>::value)
            return "int8";
        if (std::is_same<dType, uint8>::value)
            return "uint8";
        if (std::is_same<dType, int16>::value)
            return "int16";
        if (std::is_same<dType, uint16>::value)
            return "uint16";
        if (std::is_same<dType, uint32>::value)
            return "uint32";
        if (std::is_same<dType, uint64>::value)
            return "uint64";
    }

    template<Calculable dType>
    static constexpr CalculableLevel getCalculableLevel() {
        ///COMMON_USE
        if (std::is_same<dType, int32>::value)
            return int32_level;
        if (std::is_same<dType, int64>::value)
            return int64_level;
        if (std::is_same<dType, float32>::value)
            return float32_level;
        if (std::is_same<dType, float64>::value)
            return float64_level;
        if (std::is_same<dType, complex64>::value)
            return complex64_level;
        if (std::is_same<dType, complex128>::value)
            return complex128_level;

        ///RARE_USE
        if (std::is_same<dType, int8>::value)
            return int8_level;
        if (std::is_same<dType, uint8>::value)
            return uint8_level;
        if (std::is_same<dType, int16>::value)
            return int16_level;
        if (std::is_same<dType, uint16>::value)
            return uint16_level;
        if (std::is_same<dType, uint32>::value)
            return uint32_level;
        if (std::is_same<dType, uint64>::value)
            return uint64_level;
    }

    template<Calculable dType>
    static constexpr bool isComplex() {
        if (std::is_same<dType, complex64>::value)
            return true;
        if (std::is_same<dType, complex128>::value)
            return true;
        return false;
    }

    template<Calculable dType>
    static constexpr bool isReal() {
        return !isComplex<dType>();
    }

    template<Calculable dType>
    static constexpr bool isFloat() {
        if (std::is_same<dType, float32>::value)
            return true;
        if (std::is_same<dType, float64>::value)
            return true;
        return false;
    }

    template<Calculable dType>
    static constexpr bool isInt() {
        if (std::is_same<dType, int8>::value)
            return true;
        if (std::is_same<dType, int16>::value)
            return true;
        if (std::is_same<dType, int32>::value)
            return true;
        if (std::is_same<dType, int64>::value)
            return true;
        if (std::is_same<dType, uint8>::value)
            return true;
        if (std::is_same<dType, uint16>::value)
            return true;
        if (std::is_same<dType, uint32>::value)
            return true;
        if (std::is_same<dType, uint64>::value)
            return true;
        return false;
    }

    template<Calculable T1, Calculable T2>
    static constexpr bool typeHigher() {
        if (typeid(T1) == typeid(T2))
            return false;
        CalculableLevel level_t1 = getCalculableLevel<T1>();
        CalculableLevel level_t2 = getCalculableLevel<T2>();
        return level_t1 > level_t2;
    }

    template<UIntType UIT>
    static constexpr auto typeMax() -> uint_max{
        return 0.;
    }

    template<SIntType SIT>
    static constexpr auto typeMax() -> int_max{
        return 0.;
    }

    template<FloatType FT>
    static constexpr auto typeMax() -> float_max{
        return 0.;
    }

    template<ComplexType CT>
    static constexpr auto typeMax() -> complex_max{
        return 0.;
    }

    template<RealType RT>
    static constexpr auto layerMax() -> real_max{
        return 0.;
    }

    template<ComplexType CT>
    static constexpr auto layerMax() -> complex_max{
        return 0.;
    }


    static constexpr auto I(float64 imag) -> complex128 {
        return {0, imag};
    }

    static constexpr auto J(float64 imag) -> complex128 {
        return {0, imag};
    }

    static auto eps(float32 x) -> float32 {
        return x - std::nextafterf(x, std::numeric_limits<float32>::epsilon());
    }

    static auto eps(float64 x) -> float64 {
        return x - std::nextafter(x, std::numeric_limits<float64>::epsilon());
    }
}

#endif //MATSAR_TYPEUTIL_HPP
