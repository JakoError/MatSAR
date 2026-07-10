// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// check_fft.cpp -- runnable VALUE verification for the FFT family in
// MatSAR/math/DigitalSignal.hpp. This is the first end-to-end exercise of the
// butterfly (Internal::FFT): it proves the transform actually computes correctly,
// not just that it compiles.
//
// Coverage:
//   * Round-trip  ifft(fft(x)) ~= x  (within 1e-6) for
//       - N = 2, 4, 8  -> radix-2 butterfly branch (fft_loop / ipiv / merge)
//       - N = 3, 5, 6  -> non-power-of-2 direct-DFT fallback branch (if (N & (N-1)))
//     for BOTH a real input signal and a complex input signal.
//   * Known spectra checked against hand-computed values (within 1e-6):
//       - fft([1,1,1,1])   == [4,0,0,0]
//       - fft([1,0,0,0])   == [1,1,1,1]
//       - fft([1,-1,1,-1]) == [0,0,4,0]  (all energy in the Nyquist bin)
//
// Prints PASS/FAIL + max error per check; returns non-zero from main on any failure.
#include "MatSAR/math/DigitalSignal.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

using namespace matsar;

static complex128 to_c(float64 v) { return complex128(v, 0); }
static complex128 to_c(complex128 v) { return v; }

// ifft(fft(x)) ~= x within 1e-6, for a real or complex 1-D signal x.
template<typename T>
static bool roundtrip(const char *label, const Matrix<T> &x) {
    auto X = Math::fft(x);       // Matrix<complex128>
    auto xr = Math::ifft(X);     // Matrix<complex128>
    double e = 0;
    for (msize_type i = 0; i < x.getSize(); ++i)
        e = std::max(e, std::abs(xr(i) - to_c(x(i))));
    bool ok = e <= 1e-6;
    std::printf("  %-46s max_err=%.3e  %s\n", label, e, ok ? "PASS" : "FAIL");
    return ok;
}

// fft(x) == want (elementwise) within 1e-6.
static bool spectrum(const char *label, const Matrix<complex128> &got,
                     std::initializer_list<complex128> want) {
    double e = 0;
    msize_type i = 0;
    for (auto w: want) {
        e = std::max(e, std::abs(got(i) - w));
        ++i;
    }
    bool ok = e <= 1e-6;
    std::printf("  %-46s max_err=%.3e  %s\n", label, e, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;

    std::printf("== Round-trip ifft(fft(x)) ~= x (real input) ==\n");
    // Radix-2 butterfly branch (N power of 2).
    ok &= roundtrip("N=2  real  [1,2]",              Matrix<float64>({1, 2}));
    ok &= roundtrip("N=4  real  [1,2,3,4]",          Matrix<float64>({1, 2, 3, 4}));
    ok &= roundtrip("N=8  real  [1..8]",             Matrix<float64>({1, 2, 3, 4, 5, 6, 7, 8}));
    // Direct-DFT fallback branch (N not power of 2).
    ok &= roundtrip("N=3  real  [1,2,3]",            Matrix<float64>({1, 2, 3}));
    ok &= roundtrip("N=5  real  [2,4,6,8,10]",       Matrix<float64>({2, 4, 6, 8, 10}));
    ok &= roundtrip("N=6  real  [1,2,3,4,5,6]",      Matrix<float64>({1, 2, 3, 4, 5, 6}));

    std::printf("== Round-trip ifft(fft(x)) ~= x (complex input) ==\n");
    ok &= roundtrip("N=2  cplx", Matrix<complex128>({complex128(1, -1), complex128(2, 3)}));
    ok &= roundtrip("N=4  cplx", Matrix<complex128>({complex128(1, 1), complex128(2, -1),
                                                     complex128(0, 3), complex128(-1, 2)}));
    ok &= roundtrip("N=8  cplx", Matrix<complex128>({complex128(1, 0), complex128(-2, 1),
                                                     complex128(3, -3), complex128(0, 4),
                                                     complex128(-1, -1), complex128(5, 2),
                                                     complex128(2, 0), complex128(-4, 3)}));
    ok &= roundtrip("N=3  cplx", Matrix<complex128>({complex128(1, 2), complex128(-3, 1),
                                                     complex128(0, -2)}));
    ok &= roundtrip("N=5  cplx", Matrix<complex128>({complex128(1, 0), complex128(2, -1),
                                                     complex128(-1, 1), complex128(3, 2),
                                                     complex128(0, -4)}));
    ok &= roundtrip("N=6  cplx", Matrix<complex128>({complex128(2, 1), complex128(-1, 0),
                                                     complex128(0, 3), complex128(4, -2),
                                                     complex128(-3, 1), complex128(1, 1)}));

    std::printf("== Known spectra (hand-computed) ==\n");
    // fft([1,1,1,1]) = [4,0,0,0] (DC only)
    ok &= spectrum("fft([1,1,1,1]) == [4,0,0,0]",
                   Math::fft(Matrix<float64>({1, 1, 1, 1})),
                   {complex128(4, 0), complex128(0, 0), complex128(0, 0), complex128(0, 0)});
    // fft([1,0,0,0]) = [1,1,1,1] (unit impulse -> flat spectrum)
    ok &= spectrum("fft([1,0,0,0]) == [1,1,1,1]",
                   Math::fft(Matrix<float64>({1, 0, 0, 0})),
                   {complex128(1, 0), complex128(1, 0), complex128(1, 0), complex128(1, 0)});
    // fft([1,-1,1,-1]) = [0,0,4,0] (all energy in the Nyquist bin, index N/2=2)
    ok &= spectrum("fft([1,-1,1,-1]) == [0,0,4,0] (Nyquist)",
                   Math::fft(Matrix<float64>({1, -1, 1, -1})),
                   {complex128(0, 0), complex128(0, 0), complex128(4, 0), complex128(0, 0)});

    if (!ok) {
        std::printf("RESULT: FAIL\n");
        return 1;
    }
    std::printf("RESULT: PASS (all FFT value checks)\n");
    return 0;
}
