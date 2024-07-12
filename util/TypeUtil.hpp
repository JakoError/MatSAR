//
// Created by JakoError on 2022/7/2.
//
#pragma once
#ifndef MATSAR_TYPEUTIL_HPP
#define MATSAR_TYPEUTIL_HPP

#include <cmath>
#include <complex>
#include <string>
#include <limits>
#include <type_traits>

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

// define for Global
namespace MatSAR {
    // meta float type
    template<typename>
    struct float_type_helper
            : public std::false_type {
    };

    template<>
    struct float_type_helper<float32>
            : public std::true_type {
    };

    template<>
    struct float_type_helper<float64>
            : public std::true_type {
    };

    template<typename dType>
    struct float_type
            : public float_type_helper<dType>::type {
    };

    // meta complex type
    template<typename>
    struct complex_type_helper
            : public std::false_type {
    };

    template<>
    struct complex_type_helper<complex64>
            : public std::true_type {
    };

    template<>
    struct complex_type_helper<complex128>
            : public std::true_type {
    };

    template<typename dType>
    struct complex_type
            : public complex_type_helper<dType>::type {
    };

    // meta sint type
    template<typename>
    struct sint_type_helper
            : public std::false_type {
    };

    template<>
    struct sint_type_helper<int8>
            : public std::true_type {
    };

    template<>
    struct sint_type_helper<int16>
            : public std::true_type {
    };

    template<>
    struct sint_type_helper<int32>
            : public std::true_type {
    };

    template<>
    struct sint_type_helper<int64>
            : public std::true_type {
    };

    template<typename dType>
    struct sint_type
            : public sint_type_helper<dType>::type {
    };

    // meta uint type
    template<typename>
    struct uint_type_helper
            : public std::false_type {
    };

    template<>
    struct uint_type_helper<uint8>
            : public std::true_type {
    };

    template<>
    struct uint_type_helper<uint16>
            : public std::true_type {
    };

    template<>
    struct uint_type_helper<uint32>
            : public std::true_type {
    };

    template<>
    struct uint_type_helper<uint64>
            : public std::true_type {
    };

    template<typename dType>
    struct uint_type
            : public uint_type_helper<dType>::type {
    };

    // meta int type
    template<typename dType>
    struct int_type
            : public std::conditional<sint_type<dType>::value || uint_type<dType>::value,
                    std::true_type, std::false_type>::type {
    };

    // meta real type
    template<typename dType>
    struct real_type
            : public std::conditional<int_type<dType>::value || float_type<dType>::value,
                    std::true_type, std::false_type>::type {
    };

    // meta real type
    template<typename dType>
    struct calculable_type
            : public std::conditional<real_type<dType>::value || complex_type<dType>::value,
                    std::true_type, std::false_type>::type {
    };



#if (__cplusplus >= 202002L)

//    template<typename dType>
//    concept Calculable = std::is_same<dType, int8>::value ||
//                         std::is_same<dType, int16>::value ||
//                         std::is_same<dType, int32>::value ||
//                         std::is_same<dType, int64>::value ||
//                         std::is_same<dType, uint8>::value ||
//                         std::is_same<dType, uint16>::value ||
//                         std::is_same<dType, uint32>::value ||
//                         std::is_same<dType, uint64>::value ||
//                         std::is_same<dType, float32>::value ||
//                         std::is_same<dType, float64>::value ||
//                         std::is_same<dType, complex64>::value ||
//                         std::is_same<dType, complex128>::value;
//
//
//    template<typename dType>
//    concept IntType = std::is_same<dType, int8>::value ||
//                      std::is_same<dType, int16>::value ||
//                      std::is_same<dType, int32>::value ||
//                      std::is_same<dType, int64>::value ||
//                      std::is_same<dType, uint8>::value ||
//                      std::is_same<dType, uint16>::value ||
//                      std::is_same<dType, uint32>::value ||
//                      std::is_same<dType, uint64>::value;
//
//    template<typename dType>
//    concept SIntType = std::is_same<dType, int8>::value ||
//                       std::is_same<dType, int16>::value ||
//                       std::is_same<dType, int32>::value ||
//                       std::is_same<dType, int64>::value;
//
//    template<typename dType>
//    concept UIntType = std::is_same<dType, uint8>::value ||
//                       std::is_same<dType, uint16>::value ||
//                       std::is_same<dType, uint32>::value ||
//                       std::is_same<dType, uint64>::value;
//
//    template<typename dType>
//    concept FloatType = std::is_same<dType, float32>::value ||
//                        std::is_same<dType, float64>::value;
//
//    template<typename dType>
//    concept RealType = std::is_same<dType, int8>::value ||
//                       std::is_same<dType, int16>::value ||
//                       std::is_same<dType, int32>::value ||
//                       std::is_same<dType, int64>::value ||
//                       std::is_same<dType, uint8>::value ||
//                       std::is_same<dType, uint16>::value ||
//                       std::is_same<dType, uint32>::value ||
//                       std::is_same<dType, uint64>::value ||
//                       std::is_same<dType, float32>::value ||
//                       std::is_same<dType, float64>::value;
//
//    template<typename dType>
//    concept ComplexType = std::is_same<dType, complex64>::value ||
//                          std::is_same<dType, complex128>::value;

    template<typename dType>
    concept Calculable = calculable_type<dType>::value;

    template<typename dType>
    concept IntType = int_type<dType>::value;

    template<typename dType>
    concept SIntType = sint_type<dType>::value;

    template<typename dType>
    concept UIntType = uint_type<dType>::value;

    template<typename dType>
    concept FloatType = float_type<dType>::value;

    template<typename dType>
    concept RealType = real_type<dType>::value;

    template<typename dType>
    concept ComplexType = complex_type<dType>::value;

    #define CheckType(dType, Type)

#else

    # define Calculable typename
    # define IntType typename
    # define SIntType typename
    # define UIntType typename
    # define FloatType typename
    # define RealType typename
    # define ComplexType typename

    #define CheckType(dType, Type) ,class = typename std::enable_if<MatSAR::Type<dType>::value>::type

#endif

    #define CheckCalculable(dType) CheckType(dType, calculable_type)
    #define CheckInt(dType) CheckType(dType, int_type)
    #define CheckSInt(dType) CheckType(dType, sint_type)
    #define CheckUInt(dType) CheckType(dType, uint_type)
    #define CheckFloat(dType) CheckType(dType, float_type)
    #define CheckReal(dType) CheckType(dType, real_type)
    #define CheckComplex(dType) CheckType(dType, complex_type)

    enum CalculableLevel {
        unknown = -1,
        int8_level, uint8_level,
        int16_level, uint16_level,
        int32_level, uint32_level,
        int64_level, uint64_level,
        float32_level, float64_level,
        complex64_level, complex128_level
    };


}

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

namespace MatSAR {
    namespace Util {
        namespace TypeUtil {
            template<Calculable dType>
            static auto getCalculableString() -> string {
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

                if (std::is_same<dType, bool>::value)
                    return "bool";
                return "unknown";
            }

            template<Calculable dType>
            static auto getCalculableLevel() -> CalculableLevel {
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
                return unknown;
            }

            template<Calculable dType>
            static constexpr auto isComplex() -> bool {
                return std::is_same<dType, complex64>::value || std::is_same<dType, complex128>::value;
            }

            template<Calculable dType>
            static constexpr auto isReal() -> bool {
                return !isComplex<dType>();
            }

            template<Calculable dType>
            static constexpr auto isFloat() -> bool {
                return std::is_same<dType, float32>::value || std::is_same<dType, float64>::value;
            }

            template<Calculable dType>
            static constexpr auto isInt() -> bool {
                return std::is_same<dType, int8>::value ||
                       std::is_same<dType, int16>::value ||
                       std::is_same<dType, int32>::value ||
                       std::is_same<dType, int64>::value ||
                       std::is_same<dType, uint8>::value ||
                       std::is_same<dType, uint16>::value ||
                       std::is_same<dType, uint32>::value ||
                       std::is_same<dType, uint64>::value;
            }

            template<Calculable T1, Calculable T2>
            static auto typeHigher() -> bool {
                if (typeid(T1) == typeid(T2))
                    return false;
                return getCalculableLevel<T1>() > getCalculableLevel<T2>();
            }

            //    template<UIntType UIT>
            //    static  auto typeMax() -> uint_max {
            //        return 0.;
            //    }
            //
            //    template<SIntType SIT>
            //    static  auto typeMax() -> int_max {
            //        return 0.;
            //    }
            //
            //    template<FloatType FT>
            //    static  auto typeMax() -> float_max {
            //        return 0.;
            //    }
            //
            //    template<ComplexType CT>
            //    static  auto typeMax() -> complex_max {
            //        return 0.;
            //    }
            //
            //    template<RealType RT>
            //    static  auto layerMax() -> real_max {
            //        return 0.;
            //    }
            //
            //    template<ComplexType CT>
            //    static  auto layerMax() -> complex_max {
            //        return 0.;
            //    }

            //    static  auto I(float32 imag) -> complex64 {
            //        return {0, imag};
            //    }
            //
            //    static  auto J(float32 imag) -> complex64 {
            //        return {0, imag};
            //    }
            static inline auto I(float64 real, float64 imag) -> complex128 {
                return {real, imag};
            }

            static inline auto J(float64 real, float64 imag) -> complex128 {
                return {real, imag};
            }

            static inline auto I(float64 imag) -> complex128 {
                return {0, imag};
            }

            static inline auto J(float64 imag) -> complex128 {
                return {0, imag};
            }

            static inline auto eps(float32 x) -> float32 {
                return x - std::nextafterf(x, std::numeric_limits<float32>::epsilon());
            }

            static inline auto eps(float64 x) -> float64 {
                return x - std::nextafter(x, std::numeric_limits<float64>::epsilon());
            }
        }
    }
}

#endif //MATSAR_TYPEUTIL_HPP
