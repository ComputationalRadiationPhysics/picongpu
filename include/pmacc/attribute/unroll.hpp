/* Copyright 2021-2023 Bernhard Manfred Gruber
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#ifndef PMACC_UNROLL
#    define PMACC_PRAGMA(x) _Pragma(#    x)

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
