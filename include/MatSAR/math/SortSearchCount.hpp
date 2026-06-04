// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2024/8/17.
//

#ifndef MATSAR_SORTSEARCHCOUNT_HPP
#define MATSAR_SORTSEARCHCOUNT_HPP

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

#ifndef MATSAR_UNIVERSALFUNCTIONS_HPP

#include "../UniversalFunctions.hpp"

#endif //MATSAR_UNIVERSALFUNCTIONS_HPP

namespace matsar {
    namespace Math {
        namespace Internal {
            template<typename ITER_SRC_T, typename ITER_DST_T, typename WORT_T=void>
            class SORT_UP {
            public:
                void operator()(ITER_SRC_T src_iter, ITER_DST_T dst_iter, const msize_type &n,
                                __attribute__((unused)) void *work) {
                    std::copy_n(src_iter, n, dst_iter);
                    std::sort(dst_iter, dst_iter + n);
                }
            };

            template<typename ITER_SRC_T, typename ITER_DST_T, typename WORT_T=void>
            class SORT_DOWN {
            public:
                void operator()(ITER_SRC_T src_iter, ITER_DST_T dst_iter, const msize_type &n,
                                __attribute__((unused)) void *work) {
                    std::copy_n(src_iter, n, dst_iter);
                    std::sort(dst_iter, dst_iter + n);
                    std::reverse(dst_iter, dst_iter + n);
                }
            };

            template<typename dType, typename TargetType>
            auto do_global_sort(const Matrix<dType> &m, const bool &up) -> Matrix<TargetType> {
                if (up) {
                    return matsar::apply_tfunction_global_all_k<dType, TargetType, dType, Internal::SORT_UP>(m,
                                                                                                             m.getSize());
                } else {
                    return matsar::apply_tfunction_global_all_k<dType, TargetType, dType, Internal::SORT_DOWN>(m,
                                                                                                               m.getSize());
                }
            }

            template<typename dType, typename TargetType>
            auto
            do_sort(const Matrix<dType> &m, const msize_type &dim_idx, const bool &up) -> Matrix<TargetType> {
                if (up) {
                    return matsar::apply_tfunction_all_k<dType, TargetType, dType, Internal::SORT_UP>(m,
                                                                                               dim_idx,
                                                                                               m.getShape()[dim_idx]);
                } else {
                    return matsar::apply_tfunction_all_k<dType, TargetType, dType, Internal::SORT_DOWN>(m,
                                                                                                 dim_idx,
                                                                                                 m.getShape()[dim_idx]);
                }
            }
        }

        /**
         * @param dim_idx Global evaluation is performed when it is none.
         * @param up true ascending, false descending
         */
        template<typename dType>
        auto sort(const Matrix<dType> &m, Container<msize_type> dim_idx = 0, bool up = true) -> Matrix<dType> {
            if (dim_idx == none || m.isEmpty()) {
                return Internal::do_global_sort<dType, dType>(m, up);
            }
            Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), __FUNCTION__);
            return Internal::do_sort<dType, dType>(m, dim_idx, up);
        }
    }

}

#endif //MATSAR_SORTSEARCHCOUNT_HPP
