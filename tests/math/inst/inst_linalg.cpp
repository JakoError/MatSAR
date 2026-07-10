#include "MatSAR/math/LinearAlgebra.hpp"
#include <complex>
#include <vector>
#include <tuple>
#include <type_traits>
using namespace matsar;
void inst_linalg() {
    // lu: tuple<Matrix<Out>, Matrix<Out>>
    { auto r = Math::lu(eye<float32>(2));   static_assert(std::is_same_v<std::decay_t<decltype(std::get<0>(r))>, Matrix<float64>>,    "lu f32 L->f64"); }
    { auto r = Math::lu(eye<complex64>(2)); static_assert(std::is_same_v<std::decay_t<decltype(std::get<0>(r))>, Matrix<complex128>>, "lu c64 L->c128"); }
    { auto r = Math::lu(eye<float64>(2));   static_assert(std::is_same_v<std::decay_t<decltype(std::get<1>(r))>, Matrix<float64>>,    "lu f64 U keep"); }
    // lup: tuple<Matrix<Out>, Matrix<Out>, Matrix<int64>>
    { auto r = Math::lup(eye<float32>(2));   static_assert(std::is_same_v<std::decay_t<decltype(std::get<0>(r))>, Matrix<float64>>,    "lup f32->f64"); }
    { auto r = Math::lup(eye<complex64>(2)); static_assert(std::is_same_v<std::decay_t<decltype(std::get<1>(r))>, Matrix<complex128>>, "lup c64->c128"); }
    { auto r = Math::lup(eye<float64>(2));   static_assert(std::is_same_v<std::decay_t<decltype(std::get<2>(r))>, Matrix<int64>>,      "lup P int64"); }
    // det: Matrix<Out>
    { auto r = Math::det(eye<float32>(2));   static_assert(std::is_same_v<decltype(r), Matrix<float64>>,    "det f32->f64"); }
    { auto r = Math::det(eye<complex64>(2)); static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "det c64->c128"); }
    // inv: Matrix<Out>
    { auto r = Math::inv(eye<int32>(2));     static_assert(std::is_same_v<decltype(r), Matrix<float64>>,    "inv int->f64"); }
    { auto r = Math::inv(eye<float32>(2));   static_assert(std::is_same_v<decltype(r), Matrix<float64>>,    "inv f32->f64"); }
    { auto r = Math::inv(eye<complex64>(2)); static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "inv c64->c128"); }
    { auto r = Math::inv<float32>(eye<float64>(2)); static_assert(std::is_same_v<decltype(r), Matrix<float32>>, "inv override<f32> (solves f64, casts result)"); }
}
