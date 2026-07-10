#include "MatSAR/math/DigitalSignal.hpp"
#include <complex>
#include <type_traits>
using namespace matsar;
void inst_fft() {
    { auto r = Math::fft(eye<float32>(2));    static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "fft f32->c128"); }
    { auto r = Math::fft(eye<complex64>(2));  static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "fft c64->c128"); }
    { auto r = Math::fft<complex64>(eye<float64>(2)); static_assert(std::is_same_v<decltype(r), Matrix<complex64>>, "fft override<c64>"); }
    { auto r = Math::fftn(eye<float32>(2));   static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "fftn f32->c128"); }
    { auto r = Math::fft2(eye<float32>(2));   static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "fft2 f32->c128"); }
    { auto r = Math::ifft(eye<float32>(2));   static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "ifft f32->c128"); }
    { auto r = Math::ifftn(eye<float32>(2));  static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "ifftn f32->c128"); }
    { auto r = Math::ifft2(eye<float32>(2));  static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "ifft2 f32->c128"); }
}
