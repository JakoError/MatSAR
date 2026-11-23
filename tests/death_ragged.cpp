// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
// Death tests: a ragged nested initializer list or vector (rows of different lengths) is
// rejected with a ValueError instead of being copied past the end of the matrix buffer.
// Each case is selected by argv[1]; see tests/CMakeLists.txt, death_ragged_*.
#include <MatSAR/Matrix.hpp>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace matsar;
using std::vector;

int main(int argc, char **argv) {
    const char *c = argc > 1 ? argv[1] : "";
    if (!std::strcmp(c, "list_long_row")) {
        Matrix<float64> m = {{1.0, 2.0}, {3.0, 4.0, 5.0}};
    } else if (!std::strcmp(c, "list_short_row")) {
        Matrix<float64> m = {{1.0, 2.0, 3.0}, {4.0, 5.0}};
    } else if (!std::strcmp(c, "list_3d")) {
        Matrix<float64> m = {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0}, {6.0}}};   // each block is even on its own
    } else if (!std::strcmp(c, "from_rows")) {
        Matrix<float64>::fromRows({{1.0, 2.0}, {3.0}});
    } else if (!std::strcmp(c, "from_columns")) {
        Matrix<float64>::fromColumns({{1.0, 2.0}, {3.0}});
    } else if (!std::strcmp(c, "vec_2d")) {
        Matrix<float64> m(vector<vector<float64>>{{1.0, 2.0}, {3.0}});
    } else if (!std::strcmp(c, "vec_3d")) {
        Matrix<float64> m(vector<vector<vector<float64>>>{{{1.0, 2.0}, {3.0, 4.0}}, {{5.0}, {6.0}}});
    } else if (!std::strcmp(c, "from_rows_vec")) {
        Matrix<float64>::fromRows(vector<vector<float64>>{{1.0, 2.0}, {3.0}});
    } else if (!std::strcmp(c, "from_columns_vec")) {
        Matrix<float64>::fromColumns(vector<vector<float64>>{{1.0}, {2.0, 3.0}});
    }
    std::printf("no error for case '%s'\n", c);
    return 0;
}
