#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <iostream>
#include <string>

#include <Matrix.hpp>
#include <MatrixCreation.hpp>

using namespace MatSAR;
using std::string;

// HOW TO TEST PRINTING
// --------------------
// toString()/toRowColString() RETURN a std::string (callers do
// `cout << m.toString(...)`). So to *test* printing, capture the returned
// string and assert on its content. Do NOT assert exact spacing/padding --
// that is formatting, not behaviour. Check that the label, the type/shape
// annotations, and every value appear.
static bool has(const string &s, const string &sub) {
    return s.find(sub) != string::npos;
}

TEST_CASE("matrix constructs and indexes (column-major)") {
    Matrix<int64> a = {{1, 2}, {3, 4}};   // columns [1,2],[3,4] -> a(0,1)==3
    CHECK(a.getShape()[0] == 2);
    CHECK(a.getShape()[1] == 2);
    CHECK(a(0, 0) == 1);
    CHECK(a(1, 1) == 4);
    std::cout << a.toString("a") << std::endl;  // visible with `-s` or when run directly
}

TEST_CASE("toString includes name, dtype, shape, and every value") {
    Matrix<int64> a = {{11, 22}, {33, 44}};
    const string s = a.toString("myMat");
    CHECK(has(s, "myMat"));   // the label passed in
    CHECK(has(s, "int64"));   // element-type annotation
    CHECK(has(s, "shape"));   // shape annotation
    CHECK(has(s, "2x2"));
    CHECK(has(s, "11"));
    CHECK(has(s, "22"));
    CHECK(has(s, "33"));
    CHECK(has(s, "44"));
}

TEST_CASE("toString uses a default name when none is given") {
    Matrix<int64> a = {{11, 22}, {33, 44}};
    CHECK(has(a.toString(), "Matrix"));
}

TEST_CASE("toRowColString includes the name and every value") {
    Matrix<int64> a = {{11, 22}, {33, 44}};
    const string s = a.toRowColString("grid");
    CHECK(has(s, "grid"));
    CHECK(has(s, "11"));
    CHECK(has(s, "22"));
    CHECK(has(s, "33"));
    CHECK(has(s, "44"));
    CHECK_FALSE(s.empty());
}

TEST_CASE("float matrix prints fixed-point values") {
    auto m = ones({2, 2});            // Matrix<float64> filled with 1.0
    const string s = m.toString("ones");
    CHECK(has(s, "ones"));
    CHECK(has(s, "float64"));
    CHECK(has(s, "1.0000"));          // fixed-point formatting
}

TEST_CASE("toString prints logical row-major (matches operator()/toRowColString)") {
    auto m = Matrix<int64>::fromRows({{1, 2, 3}, {4, 5, 6}});  // rows [1,2,3],[4,5,6]
    const string s = m.toString("m", /*printMessage=*/false);  // braces only, no shape header
    // Row-major reading is 1,2,3,4,5,6 -> values appear in that position order.
    // The (buggy) column-major print would be 1,4,2,5,3,6 (so "4" before "3").
    CHECK(s.find("3") < s.find("4"));   // end of row 0 precedes start of row 1
    CHECK(s.find("2") < s.find("4"));
    CHECK(s.find("4") < s.find("6"));
}
