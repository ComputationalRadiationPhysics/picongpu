/* Copyright 2021 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

//! Suggests unrolling of the directly following loop to the compiler.
//!
//! Usage:
//!  `ALPAKA_UNROLL
//!  for(...){...}`
// \TODO: Implement for other compilers.
#if BOOST_ARCH_PTX
#    define ALPAKA_UNROLL_STRINGIFY(x) #x
#    define ALPAKA_UNROLL(...) _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll __VA_ARGS__))
#elif BOOST_COMP_IBM || BOOST_COMP_SUNPRO || BOOST_COMP_HPACC
#    define ALPAKA_UNROLL_STRINGIFY(x) #x
#    define ALPAKA_UNROLL(...) _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll(__VA_ARGS__)))
#elif BOOST_COMP_PGI
#    define ALPAKA_UNROLL(...) _Pragma("unroll")
#else
#    define ALPAKA_UNROLL(...)
#endif
