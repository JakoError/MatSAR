// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 24-8-1.
//

#ifndef MATSAR_LOGICFUNCTIONS_HPP
#define MATSAR_LOGICFUNCTIONS_HPP

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

#ifndef MATSAR_UNIVERSALFUNCTIONS_HPP

#include "../UniversalFunctions.hpp"

#endif //MATSAR_UNIVERSALFUNCTIONS_HPP

namespace matsar {
    //Matrix Evaluation
    namespace Math {
        namespace Internal {
            // optimized
            template<typename ITER_SRC_T, typename ITER_DST_T, typename WORT_T=void>
            class ALL_BOOL {
            public:
                void operator()(ITER_SRC_T src_iter, ITER_DST_T dst_iter, const msize_type &n,
                                __attribute__((unused)) const msize_type &k = 1,
                                __attribute__((unused)) void *work = nullptr) {
                    *dst_iter = true;
                    for (msize_type i = 0; i < n; ++i) {
                        if (!*src_iter) {
                            *dst_iter = false;
                            break;
                        }
                    }
                }
            };

            // optimized
            template<typename ITER_SRC_T, typename ITER_DST_T, typename WORT_T=void>
            class ANY_BOOL {
            public:
                void operator()(ITER_SRC_T src_iter, ITER_DST_T dst_iter, const msize_type &n,
                                __attribute__((unused)) const msize_type &k = 1,
                                __attribute__((unused)) void *work = nullptr) {
                    *dst_iter = false;
                    for (msize_type i = 0; i < n; ++i) {
                        if (*src_iter) {
                            *dst_iter = true;
                            break;
                        }
                    }
                }
            };

            // optimized
            template<typename ITER_SRC_T, typename ITER_DST_T, typename WORT_T=void>
            class ALL_NUM {
            public:
                void operator()(ITER_SRC_T src_iter, ITER_DST_T dst_iter, const msize_type &n,
                                __attribute__((unused)) const msize_type &k = 1,
                                __attribute__((unused)) void *work = nullptr) {
                    *dst_iter = true;
                    for (msize_type i = 0; i < n; ++i) {
                        if (*src_iter == 0) {
                            *dst_iter = false;
                            break;
                        }
                    }
                }
            };

            // optimized
            template<typename ITER_SRC_T, typename ITER_DST_T, typename WORT_T=void>
            class ANY_NUM {
            public:
                void operator()(ITER_SRC_T src_iter, ITER_DST_T dst_iter, const msize_type &n,
                                __attribute__((unused)) const msize_type &k = 1,
                                __attribute__((unused)) void *work = nullptr) {
                    *dst_iter = false;
                    for (msize_type i = 0; i < n; ++i) {
                        if (*src_iter != 0) {
                            *dst_iter = true;
                            break;
                        }
                    }
                }
            };

            // optimized
            template<typename dType>
            auto do_global_all(const Matrix<dType> &m) -> Matrix<bool> {
                /*
                 * apply_function_global_reduce(m, [](const bool &r, const bool &i){return r && i;}, true);
                 */
//                    bool r = true;
//                    for (const auto &i: m) {
//                        if (i == dType(0)) {
//                            r = false;
//                            break;
//                        }
//                    }
//                    return {r};
                return apply_tfunction_global_n_k<dType, bool, dType, Internal::ALL_NUM>(m, m.getSize(), 1, 0,
                                                                                         false);
            }

            // optimized
            auto do_global_all(const Matrix<bool> &m) -> Matrix<bool> {
//                    bool r = true;
//                    for (const auto &i: m) {
//                        if (!i) {
//                            r = false;
//                            break;
//                        }
//                    }
//                    return {r};
                return apply_tfunction_global_n_k<bool, bool, bool, Internal::ALL_BOOL>(m, m.getSize(), 1, 0,
                                                                                        false);
            }

            // optimized
            template<typename dType>
            auto do_all(const Matrix<dType> &m, const msize_type &dim_idx) -> Matrix<bool> {
//                    return do_all(m != dType(0), dim_idx);
                return apply_tfunction_dim_n_k<dType, bool, dType, Internal::ALL_NUM>(m, dim_idx, m.getShape()[dim_idx],
                                                                                  1, 0);
            }

            // optimized
            auto do_all(const Matrix<bool> &m, const msize_type &dim_idx) -> Matrix<bool> {
//                    return apply_function_dim_reduce(m, [](const bool &r, const bool &i) { return r && i; }, dim_idx, true);
                return apply_tfunction_dim_n_k<bool, bool, bool, Internal::ALL_BOOL>(m, dim_idx, m.getShape()[dim_idx],
                                                                                 1, 0);
            }

            // optimized
            template<typename dType>
            auto do_global_any(const Matrix<dType> &m) -> Matrix<bool> {
//                    bool r = false;
//                    for (const auto &i: m) {
//                        if (i != dType(0)) {
//                            r = true;
//                            break;
//                        }
//                    }
//                    return {r};
                return apply_tfunction_global_n_k<dType, bool, dType, ANY_NUM>(m, m.getSize(), 1, 0, false);
            }

            // optimized
            auto do_global_any(const Matrix<bool> &m) -> Matrix<bool> {
//                    bool r = false;
//                    for (const auto &i: m) {
//                        if (i) {
//                            r = true;
//                            break;
//                        }
//                    }
//                    return {r};
                return apply_tfunction_global_n_k<bool, bool, bool, ANY_BOOL>(m, m.getSize(), 1, 0, false);
            }

            template<typename dType>
            auto do_any(const Matrix<dType> &m, const msize_type &dim_idx) -> Matrix<bool> {
//                    return do_all(m != dType(0), dim_idx);
                return apply_tfunction_dim_n_k<dType, bool, dType, Internal::ANY_NUM>(m, dim_idx, m.getShape()[dim_idx],
                                                                                  1, 0);
            }

            auto do_any(const Matrix<bool> &m, const msize_type &dim_idx) -> Matrix<bool> {
//                    return apply_function_dim_reduce(m, [](const bool &r, const bool &i) { return r || i; }, dim_idx, false);
                return apply_tfunction_dim_n_k<bool, bool, bool, Internal::ANY_BOOL>(m, dim_idx, m.getShape()[dim_idx],
                                                                                 1, 0);
            }
        }

        template<typename dType>
        auto all(const Matrix<dType> &m, Container<msize_type> dim_idx = 0) -> Matrix<bool> {
            if (dim_idx == none) {
                return Internal::do_global_all(m);
            }
            Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), __FUNCTION__);
            return Internal::do_all(m, dim_idx);
        }

        template<typename dType>
        auto any(const Matrix<dType> &m, Container<msize_type> dim_idx = 0) -> Matrix<bool> {
            if (dim_idx == none) {
                return Internal::do_global_any(m);
            }
            Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), __FUNCTION__);
            return Internal::do_any(m, dim_idx);
        }
    }
}

#endif //MATSAR_LOGICFUNCTIONS_HPP
