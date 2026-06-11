/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

// clang-format off

/** @file This file should be included in each `cpp`-file before any other boost include
 * to workaround different compiler errors triggered by boost includes.
 */

/* workaround for compile error with clang-cuda
 * boost/type_traits/is_base_and_derived.hpp:142:25: error: invalid application of 'sizeof' to an incomplete type
 * 'boost::in_place_factory_base' BOOST_STATIC_ASSERT(sizeof(B) != 0);
 *
 * https://github.com/boostorg/config/issues/406#issuecomment-928151025
 */
#include <boost/utility/in_place_factory.hpp>
#include <boost/utility/typed_in_place_factory.hpp>
#include <boost/optional/optional.hpp>

// disable warnings created from inside boost mp11 with CUDA13+
#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 13)
#    pragma nv_diag_push
#    pragma nv_diag_suppress 186
// warning #186-D: pointless comparison of unsigned integer with zero
#endif

#include <boost/mp11/detail/mp_count.hpp>

#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 13)
#    pragma nv_diag_pop
#endif

#if defined(__clang__) && defined(__CUDACC__)
// Boost.Config wrongly detects the BOOST_CUDA_VERSION with clang as CUDA compiler and disables variadic templates.
// See: https://github.com/boostorg/config/issues/297
// We also need to do this after including Boost.optional, so we do not retrigger the bug the above workaround fixes.
#    undef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#    undef BOOST_NO_VARIADIC_TEMPLATES
#endif

// clang-format on
