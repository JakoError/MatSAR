#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <MatSAR.hpp>   // must compile whether or not MATSAR_WITH_BLAS is set

using namespace MatSAR;

// Exercises only the BLAS-free core (element-wise, scalar, comparison, creation).
TEST_CASE("core works without BLAS") {
    Matrix<int64> a = {{1, 2}, {3, 4}};   // a(0,0)=1, a(1,1)=4
    auto b = a * a;                        // element-wise, no BLAS
    CHECK(b(0, 0) == 1);                   // 1*1
    CHECK(b(1, 1) == 16);                  // 4*4
    CHECK((a < (a + 1))(0, 0));            // comparison + scalar add, no BLAS
}
