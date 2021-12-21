/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>

//-----------------------------------------------------------------------------
//! Suggests unrolling of the directly following loop to the compiler.
//!
//! Usage:
//!  `ALPAKA_UNROLL
//!  for(...){...}`
// \TODO: Implement for other compilers.
#if BOOST_ARCH_PTX
#    if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#        define ALPAKA_UNROLL(...) __pragma(unroll __VA_ARGS__)
#    else
#        define ALPAKA_UNROLL_STRINGIFY(x) #        x
#        define ALPAKA_UNROLL(...) _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll __VA_ARGS__))
#    endif
#else
#    if BOOST_COMP_INTEL || BOOST_COMP_IBM || BOOST_COMP_SUNPRO || BOOST_COMP_HPACC
#        define ALPAKA_UNROLL_STRINGIFY(x) #        x
#        define ALPAKA_UNROLL(...) _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll(__VA_ARGS__)))
#    elif BOOST_COMP_PGI
#        define ALPAKA_UNROLL(...) _Pragma("unroll")
#    else
#        define ALPAKA_UNROLL(...)
#    endif
#endif
