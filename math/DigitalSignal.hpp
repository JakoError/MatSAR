//
// Created by JakoError on 2022/7/10.
//
///DSP Digital Signal Process数字信号处理
#ifndef MATSAR_DIGITALSIGNAL_HPP
#define MATSAR_DIGITALSIGNAL_HPP

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

namespace MatSAR {
    namespace Math {
        template<Calculable TargetType, Calculable T1, Calculable T2>
        auto conv(const Matrix<T1> &u, const Matrix<T2> &v, const string &mode = "full") -> Matrix<TargetType> {
            auto U = u.squeeze();
            auto V = v.squeeze();
            if (!U.isVec() || !V.isVec())
                throw_value_error("conv要求两矩阵均为向量或可squeeze到向量");
            msize_type r_size = 0;
            msize_type offset = 0;
            if (mode == "full") {
                r_size = U.getSize() + V.getSize() - 1;
                offset = 0;
            } else if (mode == "same") {
                r_size = U.getSize();
                offset = V.getSize() - 2;
            } else if (mode == "valid") {
                r_size = U.getSize() - V.getSize() + 1;
                if (r_size <= 0) return {};
                offset = V.getSize() - 1;
            } else {
                throw_value_error("conv mode只允许 full same valid 三种类型");
            }
            Matrix<TargetType> r({r_size}, 0.);
            for (msize_type i = 0; i < r_size; i++) {
                for (msize_type j = 0; j < V.getSize(); ++j) {
                    if (offset + i >= j && offset + i - j < U.getSize())
                        r(i) += U(offset + i - j) * V(V.getSize() - 1 - j);
                }
            }
            return r;
        }

        template<Calculable T1, Calculable T2>
        auto conv(const Matrix<T1> &u, const Matrix<T2> &v, const string &mode = "full") -> decltype(u * v) {
            return conv<typename decltype(u * v)::dataType, T1, T2>(u, v, mode);
        }
    }
    namespace Math {
        static auto twiddle(int64 N, int64 k) -> complex128 {
            if (k == 0) return {1, 0};
            auto val = 2 * PI / static_cast<float64>(N);
            return pow(complex128(std::cos(val), -std::sin(val)), k);
        }

        template<Calculable dType>
        static void FFT(dType *src, complex128 *dst, msize_type N, dType *work) {
            if (N <= 0) return;
            if (N == 1) {
                *dst = *src;
                return;
            }
            if (N == 2) {
                dst[0] = src[0] + src[1];
                dst[1] = src[0] - src[1];
                return;
            }
            if (N & (N - 1)) {
                //dft
                for (msize_type i = 0; i < N; ++i) {
                    auto twiddle_track = twiddle(N, 0);
                    dst[i] = 0;
                    for (msize_type j = 0; j < N; ++j) {
                        if (src[j] != 0.)
                            dst[i] += src[j] * twiddle_track;
                        twiddle_track *= twiddle(N, i);
                    }
                }
            } else {
                //fft
                //pick odd
                uninitialized_copy_n(src, N, work);
                auto half_n = N >> 1;
                for (msize_type i = 0; i < half_n; ++i) {
                    src[i] = work[i << 1];
                    src[i + half_n] = work[(i << 1) + 1];
                }
                //divide
                FFT(src, dst, half_n, work);
                FFT(src + half_n, dst + half_n, half_n, work + half_n);
                //merge
                for (msize_type i = 0; i < half_n; ++i) {
                    auto tmp = dst[i];
                    dst[i + half_n] *= -twiddle(N, i);
                    dst[i] -= dst[i + half_n];
                    dst[i + half_n] += tmp;
                }
            }
        }

        template<Calculable ConvertType, Calculable dType>
        auto fft(const Matrix<dType> &m, msize_type N, msize_type dim) -> Matrix<complex128> {
            auto dim_idx = dim - 1;
            if (dim_idx >= m.getDims())
                return {m};
            if (dim_idx < 0) {
                Matrix<ConvertType> real_m({N}, 0.);
                auto iter_m = m.begin();
                for (auto iter = real_m.begin(); iter != real_m.end() && iter_m != m.end(); ++iter, ++iter_m)
                    *iter = *iter_m;
                Matrix<complex128> r({N}, 0.);
                unique_ptr<ConvertType[]> work(new ConvertType[N]);
                FFT(real_m.begin(), r.begin(), N, work.get());
                return r;
            }
            auto shape = m.getShape();
            auto step_size = MatSAR::Util::VectorUtil::mul(shape, 0, dim_idx);
            auto unit_size = step_size * shape[dim_idx];
            auto step_num = shape[dim_idx];
            auto unit_num = m.getSize() / unit_size;

            auto r_shape = m.getShape();
            r_shape[dim_idx] = N;
            Matrix<complex128> r(r_shape, 1, m.getData());
            auto r_unit_size = step_size * N;

            auto iter_r = r.getData();
            auto iter_m = m.getData();

            unique_ptr<ConvertType[]> src(new ConvertType[N]);
            unique_ptr<complex128[]> dst(new complex128[N]);
            unique_ptr<ConvertType[]> work(new ConvertType[N]);
            for (msize_type n = 0; n < unit_num; ++n) {
                for (msize_type i = 0; i < step_size; ++i) {
                    for (msize_type j = 0; j < N; ++j) {
                        src[j] = 0;
                    }
                    for (msize_type j = 0; j < step_num && j < N; ++j) {
                        src[j] = iter_m[i + j * step_size];
                    }
                    FFT(src.get(), dst.get(), N, work.get());
                    for (msize_type j = 0; j < N; ++j) {
                        iter_r[i + j * step_size] = dst[j];
                    }
                }
                iter_m += unit_size;
                iter_r += r_unit_size;
            }
            return r;
        }

        template<RealType RT>
        auto fft(const Matrix<RT> &m, msize_type N, msize_type dim) -> Matrix<complex128> {
            return fft<real_max, RT>(m, N, dim);
        }

        template<ComplexType CT>
        auto fft(const Matrix<CT> &m, msize_type N, msize_type dim) -> Matrix<complex128> {
            return fft<complex_max, CT>(m, N, dim);
        }

        template<Calculable dType>
        auto fft(const Matrix<dType> &m, msize_type dim = -1) -> Matrix<complex128> {
            auto dim_idx = dim - 1;
            //global fft
            if (dim_idx < 0) return fft(m, m.getSize(), dim);
            //auto length fft(dim)
            if (dim_idx < m.getDims()) return fft(m, m.getShape()[dim_idx], dim);
            throw_warning("fft dim参数超出矩阵维度返回原矩阵");
            return {m};
        }

        template<Calculable dType>
        auto fft2(const Matrix<dType> &m, msize_type M, msize_type N) -> Matrix<complex128> {
            return fft(fft(m, M, 1), N, 2);
        }

        template<Calculable dType>
        auto fft2(const Matrix<dType> &m) -> Matrix<complex128> {
            return fft(fft(m, 1), 2);
        }
    }
}

#endif //MATSAR_DIGITALSIGNAL_HPP