// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 24-7-25.
//

#include <string>
#ifndef MATSAR_VALUEUTIL_H
#define MATSAR_VALUEUTIL_H

namespace matsar {
    namespace Util {
        namespace ValueUtil {
            template<typename DimType1, typename DimType2>
            void validate_dimension_index(DimType1 &dim_idx, const DimType2 &dim_len,
                                          const std::string &func_name = "(function_unknown)") {
                if (dim_idx < -dim_len || dim_idx >= dim_len) {
                    throw_out_of_range_error(
                            func_name + ": " + "dimension index out of range. dim_idx = " + std::to_string(dim_idx) +
                            " matrix dims = " +
                            std::to_string(dim_len));
                }
                if (dim_idx < 0) {
                    dim_idx = dim_idx + dim_len;
                }
            }

        }
    }
}

#endif //MATSAR_VALUEUTIL_H
