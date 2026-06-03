#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <MatSAR/Matrix.hpp>
#include <MatSAR/MatrixCreation.hpp>

using namespace MatSAR;

// diag(2,3,4) is layout-independent; a column-major-aware square product below.
TEST_CASE("mmul: diagonal * identity == diagonal") {
    Matrix<float64> m = {{2, 0, 0}, {0, 3, 0}, {0, 0, 4}};
    auto id = eye<float64>(3);
    auto p = mmul(m, id);
    CHECK(p(0, 0) == doctest::Approx(2));
    CHECK(p(1, 1) == doctest::Approx(3));
    CHECK(p(2, 2) == doctest::Approx(4));
    CHECK(p(0, 1) == doctest::Approx(0));
}

// operator% is an alias for mmul.
TEST_CASE("operator% == mmul") {
    Matrix<float64> m = {{2, 0, 0}, {0, 3, 0}, {0, 0, 4}};
    auto id = eye<float64>(3);
    auto p = m % id;
    CHECK(p(0, 0) == doctest::Approx(2));
    CHECK(p(2, 2) == doctest::Approx(4));
}
