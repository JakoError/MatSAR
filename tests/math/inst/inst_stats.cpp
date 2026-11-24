#include "MatSAR/math/Probability.hpp"
#include "MatSAR/math/LinearAlgebra.hpp"
#include <complex>
#include <type_traits>
using namespace matsar;
void inst_stats() {
    { auto r = Math::median(eye<int32>(2));    static_assert(std::is_same_v<decltype(r), Matrix<real_max>>,  "median int->real_max"); }
    { auto r = Math::median(eye<float32>(2));  static_assert(std::is_same_v<decltype(r), Matrix<float32>>,   "median f32 keep"); }
    { auto r = Math::median(eye<complex64>(2));static_assert(std::is_same_v<decltype(r), Matrix<complex64>>, "median c64 keep"); }
    { auto r = Math::mean(eye<int32>(2));      static_assert(std::is_same_v<decltype(r), Matrix<real_max>>,  "mean int->real_max"); }
    { auto r = Math::mean(eye<float32>(2));    static_assert(std::is_same_v<decltype(r), Matrix<float32>>,   "mean f32 keep"); }
    { auto r = Math::mean(eye<float64>(2));    static_assert(std::is_same_v<decltype(r), Matrix<float64>>,   "mean f64 keep"); }
    { auto r = Math::mean(eye<complex64>(2));  static_assert(std::is_same_v<decltype(r), Matrix<complex64>>, "mean c64 KEEPS complex"); }
    { auto r = Math::mean(eye<complex128>(2)); static_assert(std::is_same_v<decltype(r), Matrix<complex128>>,"mean c128 keep"); }
    { auto r = Math::var(eye<int32>(2));       static_assert(std::is_same_v<decltype(r), Matrix<real_max>>,  "var int->real_max"); }
    { auto r = Math::var(eye<float32>(2));     static_assert(std::is_same_v<decltype(r), Matrix<float32>>,   "var f32 keep"); }
    { auto r = Math::var(eye<complex64>(2));   static_assert(std::is_same_v<decltype(r), Matrix<float32>>,   "var c64->f32 REAL"); }
    { auto r = Math::var(eye<complex128>(2));  static_assert(std::is_same_v<decltype(r), Matrix<float64>>,   "var c128->f64 REAL"); }
    { auto r = Math::geomean(eye<int32>(2));   static_assert(std::is_same_v<decltype(r), Matrix<real_max>>,  "geomean int->real_max"); }
    { auto r = Math::geomean(eye<float32>(2)); static_assert(std::is_same_v<decltype(r), Matrix<float32>>,   "geomean f32 keep"); }
    { auto r = Math::harmmean(eye<int32>(2));  static_assert(std::is_same_v<decltype(r), Matrix<real_max>>,  "harmmean int->real_max"); }
    { auto r = Math::harmmean(eye<float32>(2));static_assert(std::is_same_v<decltype(r), Matrix<float32>>,   "harmmean f32 keep"); }
}
