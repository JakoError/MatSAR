// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2022/7/10.
//
///DSP Digital Signal Process
#ifndef MATSAR_DIGITALSIGNAL_HPP
#define MATSAR_DIGITALSIGNAL_HPP

#include <climits>

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

#ifndef MATSAR_MATRIXCREATION_HPP

#include "../MatrixCreation.hpp"

#endif //MATSAR_MATRIXCREATION_HPP

#ifndef MATSAR_UNIVERSALFUNCTIONS_HPP

#include "../UniversalFunctions.hpp"

#endif //MATSAR_UNIVERSALFUNCTIONS_HPP

namespace matsar {
    namespace Math {
        template<typename UT, typename VT, typename TargetType=decltype(std::declval<UT>() * std::declval<VT>())>
        auto conv(const Matrix<UT> &u, const Matrix<VT> &v, const string &mode = "full") -> Matrix<TargetType> {
            auto U = u.squeeze();
            auto V = v.squeeze();
            std::reverse(V.begin(), V.end());
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
                throw_value_error(string(__FUNCTION__) + ": mode only allow 1.full 2.same 3.valid these 3 types");
                return {};
            }
            auto r = zeros<TargetType>({r_size});
            for (msize_type i = 0; i < r_size; i++) {
                for (msize_type j = 0; j < V.getSize(); ++j) {
                    if (offset + i >= j && offset + i - j < U.getSize()) {
                        r(i) += U(offset + i - j) * V(V.getSize() - 1 - j);
                    }
                }
            }
            return r;
        }
    }
    namespace Math {
        namespace Internal {
            // hint from https://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious
            template<typename T>
            static T reverse_bit(const T &v, const uint64 &N) {
                T r = std::abs(v); // r will be reversed bits of v; first get LSB of v
                T s = sizeof(v) * CHAR_BIT - 1; // extra shift needed at end

                for (v >>= 1; v; v >>= 1) {
                    r <<= 1;
                    r |= v & 1;
                    s--;
                }
                r <<= s; // shift when v's highest bits are zero
            }

            /**
             * flag for different calculation. When calculating ifft, set flag 1. When calculating fft, set flag -1.
             *
             * The twiddle factor W_N^k = exp(flag * i * 2*pi*k/N) is produced in the caller's
             * complex compute type (TargetType). Callers in the butterfly pass their own element
             * type (SRC_T) so that every twiddle multiply/accumulate stays in a single width
             * (e.g. fft<complex64> computes twiddles in complex<float>, avoiding an ill-formed
             * complex<float> * complex<double> mixed-width multiply). The angle/sign convention is
             * unchanged: TargetType defaults to complex_max, so the complex128 path is bit-for-bit
             * identical to before.
             */
            template<typename TargetType=complex_max>
            static auto twiddle(const int64 &N, const int64 &k, const float_max &flag = -1) -> TargetType {
                if (k == 0) return {1, 0};
                using real_t = typename TargetType::value_type;
                auto val = 2 * PI / static_cast<float_max>(N);
                // Exponent is cast to TargetType's own real width so pow() returns TargetType and
                // never promotes to a wider complex (e.g. pow(complex<float>, long long) would yield
                // complex<double>). For complex128 real_t is double, so pow(complex<double>, double)
                // is identical to the previous pow(complex<double>, (long long)k).
                return pow(TargetType(std::cos(val), flag * std::sin(val)), static_cast<real_t>(k));
            }

            /**
             * Calculating FFT using Butterfly Algorithm
             */
            template<typename ITER_SRC_T, typename ITER_DST_T, typename WORT_T>
            class FFT {
            public:
                using SRC_T = std::remove_reference_t<decltype(*std::declval<ITER_SRC_T>())>;

                // maps index i to its even/odd deinterleave slot: evens -> first half, odds -> second half
                static inline msize_type cal_swap_index(const msize_type i, const msize_type N) {
                    return (i & 1) ? (i >> 1) + (N >> 1) : (i >> 1);
                }

//                void ipiv(SRC_T *src, const msize_type &N) {
//                    auto half_half_N = N >> 2;
//                    for (msize_type i = 0; i < half_half_N; ++i) {
//                        auto start = (i << 1) + 1;
//                        auto carry = src[start];
//                        auto track = cal_swap_index(start, N);
//                        std::cout << start;
//                        while (track != start) {
//                            std::swap(carry, src[track]);
//                            std::cout << "->" << track;
//                            track = cal_swap_index(track, N);
//                        }
//                        src[start] = carry;
//                        std::cout << std::endl;
//                    }
//                    std::cout << std::endl;
//                }

                // In-place even/odd deinterleave via cycle-following permutation:
                // evens land in the first half, odds in the second half.
                void ipiv(SRC_T *src, const msize_type &N) {
                    auto half_half_N = N >> 2;
                    for (msize_type i = 0; i < half_half_N; ++i) {
                        auto start = (i << 1) + 1;
                        auto carry = src[start];
                        auto track = cal_swap_index(start, N);
                        while (track != start) {
                            std::swap(carry, src[track]);
                            track = cal_swap_index(track, N);
                        }
                        src[start] = carry;
                    }
                }

                void fft_loop(SRC_T *src, ITER_DST_T dst, const msize_type &N, WORT_T work) {
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
                    //fft
                    //pick odd
                    auto half_n = N >> 1;
//                    std::copy_n(src, N, work);
//                    for (msize_type i = 0; i < half_n; ++i) {
//                        src[i] = work[i << 1];
//                        src[i + half_n] = work[(i << 1) + 1];
//                    }
                    ipiv(src, N);
                    //divide
                    fft_loop(src, dst, half_n, work);
                    fft_loop(src + half_n, dst + half_n, half_n, work);
                    //merge
                    for (msize_type i = 0; i < half_n; ++i) {
                        auto tmp = dst[i];
                        dst[i + half_n] *= -twiddle<SRC_T>(N, i);
                        dst[i] -= dst[i + half_n];
                        dst[i + half_n] += tmp;
                    }
                }

                void wrap_fft(ITER_SRC_T src, ITER_DST_T dst, const msize_type &N, WORT_T work) {
                    std::unique_ptr<SRC_T[]> src_copy(new SRC_T[N]);
                    std::copy_n(src, N, src_copy.get());
                    fft_loop(src_copy.get(), dst, N, work);
                }

                void operator()(ITER_SRC_T src, ITER_DST_T dst, const msize_type &N, WORT_T work) {
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
                            auto twiddle_track = twiddle<SRC_T>(N, 0);
                            dst[i] = 0;
                            for (msize_type j = 0; j < N; ++j) {
                                /*if (src[j] != dType(0))
                                    dst[i] += src[j] * twiddle_track;*/
                                dst[i] += src[j] * twiddle_track;
                                twiddle_track *= twiddle<SRC_T>(N, i);
                            }
                        }
                    } else {
                        wrap_fft(src, dst, N, work);
                    }
                }
            };

            template<typename dType, typename TargetType>
            auto do_global_fft(const Matrix<dType> &m, const msize_type &N) -> Matrix<TargetType> {
                return apply_tfunction_global_k<dType, TargetType, dType, Internal::FFT>(m, N, 0,
                                                                                             m.getSize() == N);
            }

            template<typename dType, typename TargetType>
            auto do_fft(const Matrix<dType> &m, const msize_type &dim_idx, const msize_type &N) -> Matrix<TargetType> {
                return apply_tfunction_dim_k<dType, TargetType, dType, Internal::FFT>(m, dim_idx, N, 0);
            }

            template<typename dType, typename TargetType>
            auto do_global_ifft(const Matrix<dType> &m, const msize_type &N) -> Matrix<TargetType> {
                return do_global_fft<dType, TargetType>(m.conj(), N).conj() / (float_max) N;
            }

            template<typename dType, typename TargetType>
            auto do_ifft(const Matrix<dType> &m, const msize_type &dim_idx, const msize_type &N) -> Matrix<TargetType> {
                return do_fft<dType, TargetType>(m.conj(), dim_idx, N).conj() / (float_max) N;
            }
        }


        template<typename TargetType = void, Calculable dType>
        auto fft(const Matrix<dType> &m,
                 Container<msize_type> dim_idx = 0,
                 Container<msize_type> N = none
        ) -> Matrix<resolve_t<TargetType, complex_max>> {
            using Out = resolve_t<TargetType, complex_max>;
            static_assert(calculable_type<Out>::value, "fft: output type must be Calculable");
            // do global fft
            if (dim_idx == none) {
                if (N == none || N == -1) {
                    N = m.getSize();
                }
                if (N <= 0) {
                    throw_value_error(
                            string(string(__FUNCTION__)) + ": Invalid parameter for N as N = " + to_string(N));
                }
                return Internal::do_global_fft<Out, Out>(m, N);
            }

            // validate dim_idx
            Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), __FUNCTION__);

            //check and auto fetch N
            if (N == none || N == -1) {
                N = m.getShape()[dim_idx];
            }
            if (N <= 0) {
                throw_value_error(string(__FUNCTION__) + ": Invalid parameter for N as N = " + to_string(N));
            }

            return Internal::do_fft<Out, Out>(m, dim_idx, N);
        }

        /**
         * fftn, High n-dimensional discrete Fourier transform
         * @TargetType due to Cpp type constraint, define TargetType to confirm the middle procedure stay same type
         */
        template<typename TargetType = void, Calculable dType>
        auto fftn(const Matrix<dType> &m,
                  Container<vector<Container<msize_type>>> dim_idxs = none,
                  Container<vector<Container<msize_type>>> Ns = none
        ) -> Matrix<resolve_t<TargetType, complex_max>> {
            using Out = resolve_t<TargetType, complex_max>;
            static_assert(calculable_type<Out>::value, "fftn: output type must be Calculable");
            if (Ns == none && dim_idxs == none) {
//                throw_value_error("fftn: parameter Ns and dim_idxs cannot be both None.");
                // auto add each dim to calculate
                dim_idxs = matsar::Util::VectorUtil::Range(m.getDims());
                Ns = m.getShape();
            }
            if (Ns != none && dim_idxs != none && Ns.data().size() != dim_idxs.data().size()) {
                throw_value_error(
                        string(__FUNCTION__) + ": parameter Ns and dim_idxs should have same length. Got Ns size = " +
                        to_string(Ns.data().size()) + " dim_idxs size = " +
                        to_string(dim_idxs.data().size()));
            }
            auto n = Ns != none ? Ns.data().size() : dim_idxs.data().size();

            // auto fill Ns or dim_idxs
            if (Ns == none) {
                // already handled Ns and dim_idxs are both none
                auto Ns_predict = vector<Container<msize_type>>(n);
                auto m_shape = m.getShape();
                for (msize_type i = 0; i < n; ++i) {
                    Ns_predict[i] = m_shape[dim_idxs[i]];
                }
                Ns = Ns_predict;
            } else {
                if (dim_idxs == none) {
                    dim_idxs = matsar::Util::VectorUtil::Range(n);
                }
            }

            auto r = Matrix<Out>(m);
            for (decltype(n) i = 0; i < n; ++i) {
                auto N = Ns[i];
                auto dim_idx = dim_idxs[i];

                // do global fft
                if (dim_idx == none) {
                    if (N == none || N == -1) {
                        N = m.getSize();
                    }
                    if (N <= 0) {
                        throw_value_error(
                                string(__FUNCTION__) + ": Invalid parameter for Ns as Ns[" + to_string(i) + "] = " +
                                to_string(N));
                    }
                    r = Internal::do_global_fft<Out, Out>(r, N);
                } else {
                    // validate dim_idx
                    Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), __FUNCTION__);

                    //check and auto fetch N
                    if (N == none || N == -1) {
                        N = m.getShape()[dim_idx];
                    }
                    if (N <= 0) {
                        throw_value_error(
                                string(__FUNCTION__) + ": Invalid parameter for Ns as Ns[" + to_string(i) + "] = " +
                                to_string(N));
                    }

                    r = Internal::do_fft<Out, Out>(r, dim_idx, N);
                }
            }
            return r;
        }

        /**
         * fft2 proxy to fftn for convenience
         */
        template<typename TargetType = void, Calculable dType>
        auto fft2(const Matrix<dType> &m,
                  Container<vector<Container<msize_type>>> dim_idxs = {0, 1},
                  Container<vector<Container<msize_type>>> Ns = none
        ) -> Matrix<resolve_t<TargetType, complex_max>> {
            using Out = resolve_t<TargetType, complex_max>;
            static_assert(calculable_type<Out>::value, "fft2: output type must be Calculable");
            // check there is only two values for Ns and dim_idxs
            if (Ns != none && Ns.data().size() != 2) {
                throw_value_error(
                        string(__FUNCTION__) + ": Invalid parameter for Ns as Ns is not only have two values Ns = " +
                        to_string(Ns.data()));
            }
            if (dim_idxs != none && dim_idxs.data().size() != 2) {
                throw_value_error(
                        string(__FUNCTION__) +
                        ": Invalid parameter for dim_idxs as dim_idxs is not only have two values dim_idxs = " +
                        to_string(dim_idxs.data()));
            }
            if (Ns == none) {
                Ns = {none, none};
            }
            if (dim_idxs == none) {
                dim_idxs = {none, none};
            }

            return fftn<TargetType>(m, dim_idxs, Ns);
        }
    }
    namespace Math {
        template<typename TargetType = void, Calculable dType>
        auto ifft(const Matrix<dType> &m,
                  Container<msize_type> dim_idx = 0,
                  Container<msize_type> N = none
        ) -> Matrix<resolve_t<TargetType, complex_max>> {
            using Out = resolve_t<TargetType, complex_max>;
            static_assert(calculable_type<Out>::value, "ifft: output type must be Calculable");
            // do global ifft
            if (dim_idx == none) {
                if (N == none || N == -1) {
                    N = m.getSize();
                }
                if (N <= 0) {
                    throw_value_error(string(__FUNCTION__) + ": Invalid parameter for N as N = " + to_string(N));
                }
                return Internal::do_global_ifft<Out, Out>(m, N);
            }

            // validate dim_idx
            Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), __FUNCTION__);

            //check and auto fetch N
            if (N == none || N == -1) {
                N = m.getShape()[dim_idx];
            }
            if (N <= 0) {
                throw_value_error(string(__FUNCTION__) + ": Invalid parameter for N as N = " + to_string(N));
            }

            return Internal::do_ifft<Out, Out>(m, dim_idx, N);
        }

        /**
         * ifftn, High n-dimensional discrete Fourier transform
         * @TargetType due to Cpp type constraint, define TargetType to confirm the middle procedure stay same type
         */
        template<typename TargetType = void, Calculable dType>
        auto ifftn(const Matrix<dType> &m,
                   Container<vector<Container<msize_type>>> dim_idxs = none,
                   Container<vector<Container<msize_type>>> Ns = none
        ) -> Matrix<resolve_t<TargetType, complex_max>> {
            using Out = resolve_t<TargetType, complex_max>;
            static_assert(calculable_type<Out>::value, "ifftn: output type must be Calculable");
            if (Ns == none && dim_idxs == none) {
//                throw_value_error("ifftn: parameter Ns and dim_idxs cannot be both None.");
                // auto add each dim to calculate
                dim_idxs = matsar::Util::VectorUtil::Range(m.getDims());
                Ns = m.getShape();
            }
            if (Ns != none && dim_idxs != none && Ns.data().size() != dim_idxs.data().size()) {
                throw_value_error(
                        string(__FUNCTION__) + ": parameter Ns and dim_idxs should have same length. Got Ns size = " +
                        to_string(Ns.data().size()) + " dim_idxs size = " +
                        to_string(dim_idxs.data().size()));
            }
            auto n = Ns != none ? Ns.data().size() : dim_idxs.data().size();

            // auto fill Ns or dim_idxs
            if (Ns == none) {
                // already handled Ns and dim_idxs are both none
                auto Ns_predict = vector<Container<msize_type>>(n);
                auto m_shape = m.getShape();
                for (msize_type i = 0; i < n; ++i) {
                    Ns_predict[i] = m_shape[dim_idxs[i]];
                }
                Ns = Ns_predict;
            } else {
                if (dim_idxs == none) {
                    dim_idxs = matsar::Util::VectorUtil::Range(n);
                }
            }

            auto r = Matrix<Out>(m);
            for (decltype(n) i = 0; i < n; ++i) {
                auto N = Ns[i];
                auto dim_idx = dim_idxs[i];

                // do global ifft
                if (dim_idx == none) {
                    if (N == none || N == -1) {
                        N = m.getSize();
                    }
                    if (N <= 0) {
                        throw_value_error(
                                string(__FUNCTION__) + ": Invalid parameter for Ns as Ns[" + to_string(i) + "] = " +
                                to_string(N));
                    }
                    r = Internal::do_global_ifft<Out, Out>(r, N);
                } else {
                    // validate dim_idx
                    Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), __FUNCTION__);

                    //check and auto fetch N
                    if (N == none || N == -1) {
                        N = m.getShape()[dim_idx];
                    }
                    if (N <= 0) {
                        throw_value_error(
                                string(__FUNCTION__) + ": Invalid parameter for Ns as Ns[" + to_string(i) + "] = " +
                                to_string(N));
                    }

                    r = Internal::do_ifft<Out, Out>(r, dim_idx, N);
                }
            }
            return r;
        }

        /**
         * ifft2 proxy to ifftn for convenience
         */
        template<typename TargetType = void, Calculable dType>
        auto ifft2(const Matrix<dType> &m,
                   Container<vector<Container<msize_type>>> dim_idxs = {0, 1},
                   Container<vector<Container<msize_type>>> Ns = none
        ) -> Matrix<resolve_t<TargetType, complex_max>> {
            using Out = resolve_t<TargetType, complex_max>;
            static_assert(calculable_type<Out>::value, "ifft2: output type must be Calculable");
            // check there is only two values for Ns and dim_idxs
            if (Ns != none && Ns.data().size() != 2) {
                throw_value_error(
                        string(__FUNCTION__) + ": Invalid parameter for Ns as Ns is not only have two values Ns = " +
                        to_string(Ns.data()));
            }
            if (dim_idxs != none && dim_idxs.data().size() != 2) {
                throw_value_error(
                        string(__FUNCTION__) +
                        ": Invalid parameter for dim_idxs as dim_idxs is not only have two values dim_idxs = " +
                        to_string(dim_idxs.data()));
            }
            if (Ns == none) {
                Ns = {none, none};
            }
            if (dim_idxs == none) {
                dim_idxs = {none, none};
            }

            return ifftn<TargetType>(m, dim_idxs, Ns);
        }

        //TODO: fftshift, ifftshift
    }
//    namespace Math {
//        template<typename T>
//        static Matrix<T> mycat(int dim, const std::vector<Matrix<T>> &matrices) {//dim= 1 2
//            dim--;
//            if (matrices.empty()) {
//                throw std::invalid_argument("No matrices to concatenate.");
//            }
//
//            auto shape = matrices[0].getShape();
//            int total_dim_size = 0;
//
//            for (const auto &matrix: matrices) {
//                auto curr_shape = matrix.getShape();
//                if (curr_shape.size() != shape.size()) {
//                    throw std::invalid_argument("All input matrices must have the same number of dimensions.");
//                }
//
//                for (int i = 0; i < shape.size(); ++i) {
//                    if (i != dim && curr_shape[i] != shape[i]) {
//                        throw std::invalid_argument("The dimensions of the input matrices must match.");
//                    }
//                }
//
//                total_dim_size += curr_shape[dim];
//            }
//
//            shape[dim] = total_dim_size;
//            Matrix<T> result(shape, 0.);
//
//            int offset = 0;
//            for (auto &matrix: matrices) {
//                int dim_size = matrix.getShape()[dim];
//                Matrix<T> matrix_copy = matrix;
//                Matrix<T> sub_matrix = slice(matrix_copy, 0, dim_size, dim + 1);
//                // Matrix<T> sub_matrix = slice(matrix, 0, dim_size - 1, dim + 1);
//                for (int i = 0; i < dim_size; ++i) {
//                    for (int j = 0; j < sub_matrix.getShape()[1 - dim]; ++j) {
//                        if (dim == 0) {
//                            result(offset + i, j) = sub_matrix(i, j);
//                        } else {
//                            result(j, offset + i) = sub_matrix(j, i);
//                        }
//                    }
//                }
//                offset += dim_size;
//            }
//
//            return result;
//        }
//
//        matsar::Matrix<complex_max> operatory(const matsar::Matrix<complex_max> &matrix, int divisor, int multi) {
//            matsar::Matrix<complex_max> result = matrix;  // create a new Matrix object, initialized as a copy of the current object
//            for (auto &element: result) {  // iterate over every element of result
//                element /= static_cast<complex_max>(divisor);  // divide the element by divisor
//                element *= static_cast<complex_max>(multi);
//            }
//            return result;  // return the result
//        }
//
//        template<typename dType>
//        auto interpft(Matrix<dType> &x, int n, int dim) -> Matrix<dType> {
//            if ((x.getShape()[0] == 1 && dim == 2) || (x.getShape()[1] == 1 && dim == 1)) {
//                Matrix<dType> result;
//                auto x_iter = x.begin();
//                if (x.getShape()[0] == 1) {
//                    result = Matrix<dType>({n, x.getShape()[1]}, .0);
//                    for (int j = 0; j < result.getShape()[0]; ++j) {
//                        for (int i = 0; i < result.getShape()[1]; ++i) {
//                            result(j, i) = *x_iter;
//                            if (++x_iter == x.end()) {
//                                x_iter = x.begin();  // if the end of x has been reached, start over from the beginning
//                            }
//                        }
//                    }
//                } else {
//                    result = Matrix<dType>({x.getShape()[0], n}, .0);
//                    for (auto &element: result) {
//                        element = *x_iter;
//                        if (++x_iter == x.end()) {
//                            x_iter = x.begin();  // if the end of x has been reached, start over from the beginning
//                        }
//                    }
//                }
//                return result;
//            }
//
//            //if(x.getShape()[0] != 1&& x.getShape()[1] != 1)
//            dim = dim == 1 ? 2 : 1;
//            // Fourier transform
//            Matrix<complex128> X = fft(x, dim);
//
//            // zero padding
//            auto m = X.getShape()[dim - 1];
//            Matrix<complex128> Xzp;
//            if (dim == 1) {
//                Xzp = zeros<typename TypeTraits<dType>::ComplexType>({n, X.getShape()[1]});
//            } else {
//                Xzp = zeros<typename TypeTraits<dType>::ComplexType>({X.getShape()[0], n});
//            }
//
//            // compute the indices of the left and right halves
//            int left_end = std::ceil(static_cast<double>(m) / 2);
//            //int right_start = left_end + (m % 2 == 0 ? 0 : 1);
//            int right_start = left_end;
//            // extract the left and right halves, and apply zero padding
//            auto X_left = slice(X, 0, left_end, dim);
//            auto X_zeros = [&]() -> Matrix<complex128> {
//                if (dim == 1) {
//                    return zeros<complex128>({n - m, X.getShape()[1]});
//                } else {
//                    return zeros<complex128>({X.getShape()[0], n - m});
//                }
//            }();
//
//
//            auto X_right = slice(X, right_start, m, dim);
//
//            Xzp = matsar::Math::mycat<complex128>(dim, {X_left, X_zeros, X_right});
//
//
//            // inverse Fourier transform
//            Matrix<complex128> y = ifft(Xzp, dim);
//
//            // multiply by the ratio of new to old length to preserve energy consistency
//            y = operatory(y, m, n);
//
//            // create the result matrix and fill it with the real part of the inverse FFT result
//            Matrix<dType> result(y.getShape(), 0.);
//            auto result_iter = result.begin();
//            for (const auto &val: y) {
//                *result_iter = std::real(val);
//                ++result_iter;
//            }
//
//            return result;
//        }
//    }

}

#endif //MATSAR_DIGITALSIGNAL_HPP