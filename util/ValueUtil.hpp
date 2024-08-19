//
// Created by JakoError on 24-7-25.
//

#ifndef MATSAR_VALUEUTIL_H
#define MATSAR_VALUEUTIL_H

namespace MatSAR {
    namespace Util {
        namespace ValueUtil {
            template<typename DimType1, typename DimType2>
            void validate_dimension_index(DimType1 &dim_idx, const DimType2 &dim_len, const string &func_name = "") {
                if (dim_idx < -dim_len || dim_idx >= dim_len)
                    throw_out_of_range_error(
                            func_name + ": " + "dimension index out of range. dim_idx = " + to_string(dim_idx) +
                            " matrix dims = " +
                            to_string(dim_len));
                if (dim_idx < 0) {
                    dim_idx = dim_idx + dim_len;
                }
            }

        }
    }
}

#endif //MATSAR_VALUEUTIL_H
