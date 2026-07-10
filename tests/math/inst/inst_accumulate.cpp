#include "MatSAR/math/MathematicalFunction.hpp"
#include <complex>
#include <type_traits>
using namespace matsar;
void inst_accumulate() {
    { auto r = Math::sum(eye<int32>(2));      static_assert(std::is_same_v<decltype(r), Matrix<int64>>,   "sum int32->int64"); }
    { auto r = Math::sum(eye<uint16>(2));     static_assert(std::is_same_v<decltype(r), Matrix<uint64>>,  "sum uint16->uint64"); }
    { auto r = Math::sum(eye<int64>(2));      static_assert(std::is_same_v<decltype(r), Matrix<int64>>,   "sum int64 keep"); }
    { auto r = Math::sum(eye<float32>(2));    static_assert(std::is_same_v<decltype(r), Matrix<float32>>, "sum f32 keep"); }
    { auto r = Math::sum(eye<float64>(2));    static_assert(std::is_same_v<decltype(r), Matrix<float64>>, "sum f64 keep"); }
    { auto r = Math::sum(eye<complex64>(2));  static_assert(std::is_same_v<decltype(r), Matrix<complex64>>,"sum c64 keep"); }
    { auto r = Math::sum<float64>(eye<int32>(2)); static_assert(std::is_same_v<decltype(r), Matrix<float64>>, "sum override<f64>"); }
    { auto r = Math::prod(eye<int32>(2));     static_assert(std::is_same_v<decltype(r), Matrix<int64>>,   "prod int32->int64"); }
    { auto r = Math::prod(eye<complex64>(2)); static_assert(std::is_same_v<decltype(r), Matrix<complex64>>,"prod c64 keep"); }
    { auto r = Math::cumsum(eye<int32>(2));   static_assert(std::is_same_v<decltype(r), Matrix<int64>>,   "cumsum int32->int64"); }
    { auto r = Math::cumsum(eye<complex64>(2));static_assert(std::is_same_v<decltype(r), Matrix<complex64>>,"cumsum c64 keep"); }
    { auto r = Math::cumprod(eye<int32>(2));  static_assert(std::is_same_v<decltype(r), Matrix<int64>>,   "cumprod int32->int64"); }
    { auto r = Math::cumprod(eye<complex64>(2));static_assert(std::is_same_v<decltype(r), Matrix<complex64>>,"cumprod c64 keep"); }
}
