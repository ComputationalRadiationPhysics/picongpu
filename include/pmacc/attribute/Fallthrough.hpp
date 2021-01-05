/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz,
 *                     Alexander Grund
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

// compatibility macros (compiler or C++ standard version specific)
#include <boost/config.hpp>
// work-around for Boost 1.68.0
//   fixed in https://github.com/boostorg/predef/pull/84
//   see https://github.com/ComputationalRadiationPhysics/alpaka/pull/606
// include <boost/predef.h>
#include <alpaka/core/BoostPredef.hpp>


/** C++11 and C++14 explicit fallthrough in switch cases
 *
 * Use [[fallthrough]] in C++17
 */
#if(BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(7, 0, 0))
#    define PMACC_FALLTHROUGH [[gnu::fallthrough]]
#elif BOOST_COMP_CLANG
#    define PMACC_FALLTHROUGH [[clang::fallthrough]]
#else
#    define PMACC_FALLTHROUGH ((void) 0)
#endif
