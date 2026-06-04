// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Created by JakoError on 2022/7/27.
//
// MatSAR public header.
// The stable core is always available. Unfinished/experimental modules are
// compiled only when MATSAR_ENABLE_EXPERIMENTAL is defined (CMake option of
// the same name). They are not deleted -- just quarantined until each is
// re-stabilized with its own tests.
//

#ifndef MATSAR_HPP
#define MATSAR_HPP

// ---- stable core ----
#include "Matrix.hpp"
#include "MatrixCreation.hpp"

// ---- experimental (opt-in) ----
// The experimental headers are excluded from the public mirror; the __has_include
// guard makes MATSAR_ENABLE_EXPERIMENTAL a clean no-op there (instead of a
// missing-header error) while still compiling in the full source tree.
#if defined(MATSAR_ENABLE_EXPERIMENTAL) && __has_include("SparseMatrix.hpp")

#include "SparseMatrix.hpp"
#include "UniversalFunctions.hpp"
#include "math/SortSearchCount.hpp"
#include "math/LogicFunctions.hpp"
#include "math/MathematicalFunction.hpp"
#include "math/Trigonometry.hpp"
#include "math/MatrixOperation.hpp"
#include "math/LinearAlgebra.hpp"
//#include "math/Probability.hpp"
#include "math/DigitalSignal.hpp"

#endif // experimental (opt-in)

#endif //MATSAR_HPP
