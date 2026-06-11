/*
 * SPDX-FileCopyrightText: Bernhard Manfred Gruber
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#ifndef PMACC_UNROLL
#    define PMACC_PRAGMA(x) _Pragma(#x)

#    if defined(__clang__) || defined(__INTEL_LLVM_COMPILER) || defined(__NVCC__)
#        define PMACC_UNROLL(var) PMACC_PRAGMA(unroll var)
#    elif defined(__INTEL_COMPILER) // check Intel before g++, because Intel defines __GNUG__
#        define PMACC_UNROLL(var) PMACC_PRAGMA(unroll(var))
#    elif defined(__GNUG__)
// g++ does support an unroll pragma but it does not accept the value of a template argument (at least until g++-11.2)
// see also: https://stackoverflow.com/q/63404539/2406044
// #define PMACC_UNROLL(var) PMACC_PRAGMA(GCC unroll var)
#        define PMACC_UNROLL(var)
#    elif defined(_MSC_VER)
// MSVC does not support a pragma for unrolling
#        define PMACC_UNROLL(var)
#    else
#        define PMACC_UNROLL(var)
#        warning PMACC_UNROLL is not implemented for your compiler
#    endif
#endif
