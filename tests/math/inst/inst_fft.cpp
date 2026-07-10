#include "MatSAR/math/DigitalSignal.hpp"
#include <complex>
#include <type_traits>
#include <utility>
using namespace matsar;
void inst_fft() {
    { auto r = Math::fft(eye<float32>(2));    static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "fft f32->c128"); }
    { auto r = Math::fft(eye<complex64>(2));  static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "fft c64->c128"); }
    // fft output override to complex64: the return-TYPE normalization is verified via decltype (signature).
    // A complex64 COMPUTE path can't fully instantiate the butterfly today because twiddle() is fixed
    // complex128, giving a mixed-width complex64*complex128 multiply (DigitalSignal.hpp:200) -> task_56e6cf57.
    // The default complex128 path (all cases above/below) fully instantiates.
    static_assert(std::is_same_v<decltype(Math::fft<complex64>(std::declval<const Matrix<float64>&>())), Matrix<complex64>>, "fft override<c64> signature");
    { auto r = Math::fftn(eye<float32>(2));   static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "fftn f32->c128"); }
    { auto r = Math::fft2(eye<float32>(2));   static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "fft2 f32->c128"); }
    { auto r = Math::ifft(eye<float32>(2));   static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "ifft f32->c128"); }
    { auto r = Math::ifftn(eye<float32>(2));  static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "ifftn f32->c128"); }
    { auto r = Math::ifft2(eye<float32>(2));  static_assert(std::is_same_v<decltype(r), Matrix<complex128>>, "ifft2 f32->c128"); }
}
