#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <Matrix.hpp>
#include <MatrixCreation.hpp>

using namespace MatSAR;

TEST_CASE("core container constructs and indexes") {
    Matrix<int64> a = {{1, 2}, {3, 4}};      // 2x2, row-major
    CHECK(a.getShape()[0] == 2);
    CHECK(a.getShape()[1] == 2);
    CHECK(a(0, 0) == 1);
    CHECK(a(1, 1) == 4);
}

TEST_CASE("creation helpers") {
    auto e = eye<int64>(3);
    CHECK(e(0, 0) == 1);
    CHECK(e(0, 1) == 0);
    CHECK(e(2, 2) == 1);
}
