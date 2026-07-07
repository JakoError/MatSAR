// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
#ifndef MATSAR_TEST_RNG_HPP
#define MATSAR_TEST_RNG_HPP
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace matsar_test {
struct Rng {
    std::mt19937_64 eng;
    explicit Rng(std::uint64_t seed) : eng(seed) {}
    // precondition: lo <= hi (std distributions are UB otherwise)
    double uniform(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(eng); }
    // precondition: lo <= hi (std distributions are UB otherwise)
    std::size_t uni(std::size_t lo, std::size_t hi) { return std::uniform_int_distribution<std::size_t>(lo, hi)(eng); }
};

// A random shape of `dims` dimensions, each in [lo, hi].
inline std::vector<std::size_t> random_shape(Rng& rng, std::size_t dims, std::size_t lo, std::size_t hi) {
    std::vector<std::size_t> s(dims);
    for (auto& d : s) d = rng.uni(lo, hi);
    return s;
}
}  // namespace matsar_test
#endif
