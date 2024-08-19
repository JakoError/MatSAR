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

namespace MatSAR {
    namespace Math {
        namespace Internal {
            template<typename ITER_SRC_T, typename ITER_DST_T>
            void
            SORT_UP(ITER_SRC_T src_iter, ITER_DST_T dst_iter, const msize_type &k,
                    __attribute__((unused)) void *work) {
                std::copy_n(src_iter, k, dst_iter);
                std::sort(dst_iter, dst_iter + k);
            }

            template<typename ITER_SRC_T, typename ITER_DST_T>
            void SORT_DOWN(ITER_SRC_T src_iter, ITER_DST_T dst_iter, const msize_type &k,
                           __attribute__((unused)) void *work) {
                std::copy_n(src_iter, k, dst_iter);
                std::sort(dst_iter, dst_iter + k);
                std::reverse(dst_iter, dst_iter + k);
            }

            template<typename dType, typename TargetType>
            auto do_global_sort(const Matrix<dType> &m, const bool &up) -> Matrix<TargetType> {
                if (up) {
                    return MatSAR::apply_function_global_nk<dType, TargetType>(m,
                                                                               Internal::SORT_UP<typeof(std::declval<Matrix<dType>>().getData()), typeof(std::declval<Matrix<dType>>().getData())>,
                                                                               m.getSize());
                } else {
                    return MatSAR::apply_function_global_nk<dType, TargetType>(m,
                                                                               Internal::SORT_DOWN<typeof(std::declval<Matrix<dType>>().getData()), typeof(std::declval<Matrix<dType>>().getData())>,
                                                                               m.getSize());
                }
            }

            template<typename dType, typename TargetType>
            auto
            do_sort(const Matrix<dType> &m, const msize_type &dim_idx, const bool &up) -> Matrix<TargetType> {
                if (up) {
                    return MatSAR::apply_function_nk<dType, TargetType>(m,
                                                                        Internal::SORT_UP<typeof(std::declval<Matrix<dType>>().getData()), typeof(std::declval<Matrix<dType>>().getData())>,
                                                                        dim_idx, m.getShape()[dim_idx]);
                } else {
                    return MatSAR::apply_function_nk<dType, TargetType>(m,
                                                                        Internal::SORT_DOWN<typeof(std::declval<Matrix<dType>>().getData()), typeof(std::declval<Matrix<dType>>().getData())>,
                                                                        dim_idx, m.getShape()[dim_idx]);
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
            Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), "sort");
            return Internal::do_sort<dType, dType>(m, dim_idx, up);
        }
    }

}

#endif //MATSAR_SORTSEARCHCOUNT_HPP
