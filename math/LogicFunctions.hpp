//
// Created by JakoError on 24-8-1.
//

#ifndef MATSAR_LOGICFUNCTIONS_HPP
#define MATSAR_LOGICFUNCTIONS_HPP

#ifndef MATSAR_MATRIX_HPP

#include "../Matrix.hpp"

#endif //MATSAR_MATRIX_HPP

namespace MatSAR {
    namespace Math {
        //Matrix Evaluation
        namespace Math {
            namespace Internal {
                // optimized
                template<typename dType>
                auto do_global_all(const Matrix<dType> &m) -> Matrix<bool> {
                    /*
                     * apply_function_global(m, [](const bool &r, const bool &i){return r && i;}, true);
                     */
                    bool r = true;
                    for (const auto &i: m) {
                        if (i == dType(0)) {
                            r = false;
                            break;
                        }
                    }
                    return {r};
                }

                // optimized
                auto do_global_all(const Matrix<bool> &m) -> Matrix<bool> {
                    bool r = true;
                    for (const auto &i: m) {
                        if (!i) {
                            r = false;
                            break;
                        }
                    }
                    return {r};
                }

                template<typename dType>
                auto do_all(const Matrix<dType> &m, const msize_type &dim_idx) -> Matrix<bool> {
                    return do_all(m != dType(0), dim_idx);
                }

                auto do_all(const Matrix<bool> &m, const msize_type &dim_idx) -> Matrix<bool> {
                    return apply_function_n1(m, [](const bool &r, const bool &i) { return r && i; }, dim_idx, true);
                }

                // optimized
                template<typename dType>
                auto do_global_any(const Matrix<dType> &m) -> Matrix<bool> {
                    bool r = false;
                    for (const auto &i: m) {
                        if (i != dType(0)) {
                            r = true;
                            break;
                        }
                    }
                    return {r};
                }

                // optimized
                auto do_global_any(const Matrix<bool> &m) -> Matrix<bool> {
                    bool r = false;
                    for (const auto &i: m) {
                        if (i) {
                            r = true;
                            break;
                        }
                    }
                    return {r};
                }

                template<typename dType>
                auto do_any(const Matrix<dType> &m, const msize_type &dim_idx) -> Matrix<bool> {
                    return do_all(m != dType(0), dim_idx);
                }

                auto do_any(const Matrix<bool> &m, const msize_type &dim_idx) -> Matrix<bool> {
                    return apply_function_n1(m, [](const bool &r, const bool &i) { return r || i; }, dim_idx, false);
                }
            }

            template<typename dType>
            auto all(const Matrix<dType> &m, Container<msize_type> dim_idx = 0) -> Matrix<bool> {
                if (dim_idx == none) {
                    return Internal::do_global_all<dType>(m, dim_idx);
                }
                Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), "all");
                return Internal::do_all<dType>(m, dim_idx);
            }

            template<typename dType>
            auto any(const Matrix<dType> &m, Container<msize_type> dim_idx = 0) -> bool {
                if (dim_idx == none) {
                    return Internal::do_global_any<dType>(m, dim_idx);
                }
                Util::ValueUtil::validate_dimension_index(dim_idx, m.getDims(), "all");
                return Internal::do_any<dType>(m, dim_idx);
            }
        }
    }
}

#endif //MATSAR_LOGICFUNCTIONS_HPP
