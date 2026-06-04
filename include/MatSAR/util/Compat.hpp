// Copyright 2026 Zhexian Zhou
// SPDX-License-Identifier: Apache-2.0
//
// Portability helpers so the core compiles on MSVC as well as GCC/Clang.
//
#ifndef MATSAR_COMPAT_HPP
#define MATSAR_COMPAT_HPP

// Portable "unused" attribute.
#if defined(__GNUC__) || defined(__clang__)
#define MATSAR_UNUSED __attribute__((unused))
#else
#define MATSAR_UNUSED
#endif

// Portable warning push/pop.
#if defined(_MSC_VER)
#define MATSAR_DIAG_PUSH __pragma(warning(push))
#define MATSAR_DIAG_POP __pragma(warning(pop))
#elif defined(__clang__)
#define MATSAR_DIAG_PUSH _Pragma("clang diagnostic push")
#define MATSAR_DIAG_POP _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#define MATSAR_DIAG_PUSH _Pragma("GCC diagnostic push")
#define MATSAR_DIAG_POP _Pragma("GCC diagnostic pop")
#else
#define MATSAR_DIAG_PUSH
#define MATSAR_DIAG_POP
#endif

#endif // MATSAR_COMPAT_HPP
