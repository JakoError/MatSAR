#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <MatSAR.hpp>   // default (experimental OFF) public header must compile

using namespace MatSAR;

TEST_CASE("public header exposes the core") {
    auto m = ones({2, 2});
    CHECK(m(0, 0) == doctest::Approx(1.0));
}
