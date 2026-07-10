// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 24-8-5.
//

#ifndef MATSAR_MATHEMATICALFUNCTION_HPP
#define MATSAR_MATHEMATICALFUNCTION_HPP

#ifndef MATSAR_MATRIXCREATION_HPP

#include "../MatrixCreation.hpp"

#endif //MATSAR_MATRIXCREATION_HPP

#ifndef MATSAR_UNIVERSALFUNCTIONS_HPP

#include "../UniversalFunctions.hpp"

#endif //MATSAR_UNIVERSALFUNCTIONS_HPP

namespace matsar {
    namespace Math {
        namespace Internal {
            template<typename dType, typename TargetType>
            auto do_global_sum(const Matrix<dType> &m)
                    -> Matrix<TargetType> {
                return matsar::apply_function_global_reduce<dType, TargetType>(
                        m,
                        [](const TargetType &x, const dType &y) {
                            return static_cast<TargetType>(x + y);
                        },
                        0);
            }

            template<typename dType, typename TargetType>
            auto do_sum(const Matrix<dType> &m, const msize_type &dim_idx)
                    -> Matrix<TargetType> {
                return matsar::apply_function_dim_reduce<dType, TargetType>(
                        m, [](const TargetType &x, const dType &y) {
                            return static_cast<TargetType>(x + y);
                        },
                        dim_idx, 0);
            }
        }

        /**
         * @param dim_idx Global evaluation is performed when it is none.
         */
        template<typename TargetType = void, Calculable dType>
        auto sum(const Matrix<dType> &m, Container<msize_type> dim_idx = 0)
                -> Matrix<resolve_t<TargetType, accumulate_t<dType>>> {
            using Out = resolve_t<TargetType, accumulate_t<dType>>;
            static_assert(calculable_type<Out>::value, "sum: output type must be Calculable");
            if (dim_idx == none || m.isEmpty()) {
                return Internal::do_global_sum<dType, Out>(m);
            }
            Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), __FUNCTION__);
            return Internal::do_sum<dType, Out>(m, dim_idx);
        }
    }

    namespace Math {
        namespace Internal {
            template<typename dType, typename TargetType>
            auto do_global_prod(const Matrix<dType> &m) -> Matrix<TargetType> {
                return matsar::apply_function_global_reduce<dType, TargetType>(
                        m,
                        [](const TargetType &x, const dType &y) {
                            return static_cast<TargetType>(x * y);
                        },
                        1);
            }

            template<typename dType, typename TargetType>
            auto do_prod(const Matrix<dType> &m, const msize_type &dim_idx) -> Matrix<TargetType> {
                return matsar::apply_function_dim_reduce<dType, TargetType>(
                        m,
                        [](const TargetType &x, const dType &y) {
                            return static_cast<TargetType>(x * y);
                        },
                        dim_idx, 1);
            }
        }

        /**
         * @param dim_idx Global evaluation is performed when it is none.
         */
        template<typename TargetType = void, Calculable dType>
        auto prod(const Matrix<dType> &m, Container<msize_type> dim_idx = 0)
                -> Matrix<resolve_t<TargetType, accumulate_t<dType>>> {
            using Out = resolve_t<TargetType, accumulate_t<dType>>;
            static_assert(calculable_type<Out>::value, "prod: output type must be Calculable");
            if (dim_idx == none || m.isEmpty()) {
                return Internal::do_global_prod<dType, Out>(m);
            }
            Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), __FUNCTION__);
            return Internal::do_prod<dType, Out>(m, dim_idx);
        }
    }

    namespace Math {
        namespace Internal {
            template<typename dType, typename TargetType>
            auto do_global_cumsum(const Matrix<dType> &m) -> Matrix<TargetType> {
                TargetType sum = 0;
                return matsar::apply_function_global_map<dType, TargetType>(
                        m,
                        [&sum](const TargetType &x,
                               const dType &y) {
                            sum = static_cast<TargetType>(sum +
                                                          y);
                            return sum;
                        });
            }

            template<typename dType, typename TargetType>
            auto do_cumsum(const Matrix<dType> &m, const msize_type &dim_idx) -> Matrix<TargetType> {
                TargetType sum = 0;
                return matsar::apply_function_dim_map_idx<dType, TargetType>(
                        m, [&sum](const TargetType &x, const dType &y, const typename Matrix<dType>::size_type &idx) {
                            if (idx == 0) {
                                sum = 0;
                            }
                            sum = static_cast<TargetType>(sum + y);
                            return sum;
                        },
                        dim_idx, 0);
            }
        }

        /**
         * @param dim_idx Global evaluation is performed when it is none.
         */
        template<typename dType, typename TargetType = void>
        auto cumsum(const Matrix<dType> &m, Container<msize_type> dim_idx = 0)
                -> Matrix<std::conditional_t<
                        std::is_void_v<TargetType>,
                        remove_cvref_t<decltype(std::declval<dType>() + std::declval<dType>())>,
                        TargetType>> {
            using Out = std::conditional_t<
                    std::is_void_v<TargetType>,
                    remove_cvref_t<decltype(std::declval<dType>() + std::declval<dType>())>,
                    TargetType>;
            if (dim_idx == none || m.isEmpty()) {
                return Internal::do_global_cumsum<dType, Out>(m);
            }
            Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), __FUNCTION__);
            return Internal::do_cumsum<dType, Out>(m, dim_idx);
        }
    }

    namespace Math {
        namespace Internal {
            template<typename dType, typename TargetType>
            auto do_global_cumprod(const Matrix<dType> &m) -> Matrix<TargetType> {
                TargetType prod = 1;
                return matsar::apply_function_global_map<dType, TargetType>(
                        m, [&prod](const TargetType &x, const dType &y) {
                            prod = static_cast<TargetType>(prod * y);
                            return prod;
                        },
                        0);
            }

            template<typename dType, typename TargetType>
            auto do_cumprod(const Matrix<dType> &m, const msize_type &dim_idx) -> Matrix<TargetType> {
                TargetType prod = 1;
                return matsar::apply_function_dim_map_idx<dType, TargetType>(
                        m, [&prod](const TargetType &x, const dType &y, const typename Matrix<dType>::size_type &idx) {
                            if (idx == 0) {
                                prod = 1;
                            }
                            prod = static_cast<TargetType>(prod * y);
                            return prod;
                        },
                        dim_idx, 0);
            }
        }

        /**
         * @param dim_idx Global evaluation is performed when it is none.
         */
        template<typename dType, typename TargetType = void>
        auto cumprod(const Matrix<dType> &m, Container<msize_type> dim_idx = 0)
                -> Matrix<std::conditional_t<
                        std::is_void_v<TargetType>,
                        remove_cvref_t<decltype(std::declval<dType>() * std::declval<dType>())>,
                        TargetType>> {
            using Out = std::conditional_t<
                    std::is_void_v<TargetType>,
                    remove_cvref_t<decltype(std::declval<dType>() * std::declval<dType>())>,
                    TargetType>;
            if (dim_idx == none || m.isEmpty()) {
                return Internal::do_global_cumprod<dType, Out>(m);
            }
            Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), __FUNCTION__);
            return Internal::do_cumprod<dType, Out>(m, dim_idx);
        }
    }
}

#endif //MATSAR_MATHEMATICALFUNCTION_HPP
