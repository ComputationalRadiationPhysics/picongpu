/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Common.hpp>

//-----------------------------------------------------------------------------
//! Suggests unrolling of the directly following loop to the compiler.
//!
//! Usage:
//!  `ALPAKA_UNROLL
//!  for(...){...}`
// \TODO: Implement for other compilers.
#if BOOST_ARCH_CUDA_DEVICE
    #if BOOST_COMP_MSVC
        #define ALPAKA_UNROLL(...) __pragma(unroll __VA_ARGS__)
    #else
        #define ALPAKA_UNROLL_STRINGIFY(x) #x
        #define ALPAKA_UNROLL(...)  _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll __VA_ARGS__))
    #endif
#else
    #if BOOST_COMP_INTEL || BOOST_COMP_IBM || BOOST_COMP_SUNPRO || BOOST_COMP_HPACC
        #define ALPAKA_UNROLL_STRINGIFY(x) #x
        #define ALPAKA_UNROLL(...)  _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll(__VA_ARGS__)))
    #elif BOOST_COMP_PGI
        #define ALPAKA_UNROLL(...)  _Pragma("unroll")
    #else
        #define ALPAKA_UNROLL(...)
    #endif
#endif
