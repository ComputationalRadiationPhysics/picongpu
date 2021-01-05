/* Copyright 2020-2021 Rene Widera
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

/** @file This file should be included in each `cpp`-file before any other boost include
 * to workaround compiler errors when compiling with clang-cuda and boost <1.69.0
 *
 * https://github.com/ComputationalRadiationPhysics/picongpu/issues/3294
 */
#include <boost/version.hpp>
#if(BOOST_VERSION < 106900 && defined(__CUDACC__) && defined(__clang__))
#    if defined(__CUDACC__)
#        include <boost/config/compiler/nvcc.hpp>
#    endif
#    if(!defined(__ibmxl__))
#        include <boost/config/compiler/clang.hpp>
#    endif
#    undef __CUDACC__
#    include <boost/config/detail/select_compiler_config.hpp>
#    define __CUDACC__
#endif
/* workaround for compile error with clang-cuda
 * boost/type_traits/is_base_and_derived.hpp:142:25: error: invalid application of 'sizeof' to an incomplete type
 * 'boost::in_place_factory_base' BOOST_STATIC_ASSERT(sizeof(B) != 0);
 */
#include <boost/optional/optional.hpp>
