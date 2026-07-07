// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
#ifndef MATSAR_TEST_GOLDEN_HPP
#define MATSAR_TEST_GOLDEN_HPP
#include <cstddef>
#include <string>
#include <vector>
#include <MatSAR/Matrix.hpp>
#include <MatSAR/MatrixCreation.hpp>
#include "doctest.h"     // CHECK/REQUIRE/INFO/CAPTURE used by check_matrix
#include "approx.hpp"
#include "rng.hpp"

namespace matsar_test {
using matsar::Matrix;

inline constexpr long long AXIS_NONE = -1;

struct GoldenCase {
    const char* name;
    std::vector<double> in;   std::vector<std::size_t> in_shape;    // row-major logical
    std::vector<double> in2;  std::vector<std::size_t> in2_shape;   // 2nd input (empty if unused)
    long long           axis;                                        // >=0, or AXIS_NONE
    std::vector<double> out;  std::vector<std::size_t> out_shape;    // row-major logical
    double rtol, atol;
};

// Build a Matrix<T> from a row-major logical flat array + shape (1-D or 2-D).
template<class T>
Matrix<T> build(const std::vector<double>& flat, const std::vector<std::size_t>& shape) {
    if (shape.size() == 1) {
        std::vector<T> v; v.reserve(flat.size());
        for (double x : flat) v.push_back(static_cast<T>(x));
        return Matrix<T>(v);                        // 1-D vector ctor -> shape {n}
    }
    const std::size_t rows = shape[0], cols = shape[1];
    std::vector<std::vector<T>> vv(rows, std::vector<T>(cols));
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            vv[i][j] = static_cast<T>(flat[i * cols + j]);
    return Matrix<T>::fromRows(vv);                 // 2-D row-major logical
}

// Compare a result matrix against expected (row-major logical flat) with tolerance.
template<class T>
void check_matrix(const char* name, const Matrix<T>& got,
                  const std::vector<double>& expect, const std::vector<std::size_t>& shape,
                  double rtol, double atol) {
    INFO("case: " << std::string(name));
    auto gs = got.getShape();
    REQUIRE(gs.size() == shape.size());
    for (std::size_t d = 0; d < shape.size(); ++d) { CAPTURE(d); CHECK(static_cast<std::size_t>(gs[d]) == shape[d]); }
    if (shape.size() == 1) {
        for (std::size_t i = 0; i < shape[0]; ++i) {
            CAPTURE(i);
            CHECK(approx_eq(static_cast<double>(got(i)), expect[i], rtol, atol));
        }
    } else {
        const std::size_t rows = shape[0], cols = shape[1];
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; ++j) {
                CAPTURE(i); CAPTURE(j);
                CHECK(approx_eq(static_cast<double>(got(i, j)), expect[i * cols + j], rtol, atol));
            }
    }
}

// A random Matrix<T> for the property layer.
template<class T>
Matrix<T> random_matrix(Rng& rng, const std::vector<std::size_t>& shape, double lo, double hi) {
    std::size_t n = 1; for (auto d : shape) n *= d;
    std::vector<double> flat(n);
    for (auto& x : flat) x = rng.uniform(lo, hi);
    return build<T>(flat, shape);
}

// Oracle-free brute-force references (order-independent).
template<class T> double brute_sum(const Matrix<T>& m)  { double s = 0; for (const auto& x : m) s += static_cast<double>(x); return s; }
template<class T> double brute_prod(const Matrix<T>& m) { double p = 1; for (const auto& x : m) p *= static_cast<double>(x); return p; }
template<class T> double reduce_all(const Matrix<T>& m) { return brute_sum(m); }
}  // namespace matsar_test
#endif
