/* Copyright 2013-2019 Felix Schmitt, Heiko Burau, Rene Widera,
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


#define BOOST_MPL_LIMIT_VECTOR_SIZE 20
#define BOOST_MPL_LIMIT_MAP_SIZE 20

#include <cupla/types.hpp>

#ifndef PMACC_CUDA_ENABLED
#   define PMACC_CUDA_ENABLED ALPAKA_ACC_GPU_CUDA_ENABLED
#endif

#if( PMACC_CUDA_ENABLED == 1 )
/* include mallocMC before cupla renaming is activated, else we need the variable acc
 * to call atomic cuda functions
 */
#   include <mallocMC/mallocMC.hpp>
#endif


#include <cuda_to_cupla.hpp>

#if( PMACC_CUDA_ENABLED == 1 )
/** @todo please remove this workaround
 * This workaround allows to use native CUDA on the CUDA device without
 * passing the variable `acc` to each function. This is only needed during the
 * porting phase to allow the full feature set of the plain PMacc and PIConGPU
 * CUDA version if the accelerator is CUDA.
 */
#   undef blockIdx
#   undef __syncthreads
#   undef threadIdx
#   undef gridDim
#   undef blockDim
#   undef uint3
#   undef dim3

#endif

#include "pmacc/debug/PMaccVerbose.hpp"
#include "pmacc/ppFunctions.hpp"

#include "pmacc/dimensions/Definition.hpp"
#include "pmacc/type/Area.hpp"
#include "pmacc/type/Integral.hpp"
#include "pmacc/type/Exchange.hpp"
#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/attribute/Constexpr.hpp"
#include "pmacc/attribute/Fallthrough.hpp"
#include "pmacc/eventSystem/EventType.hpp"
#include "pmacc/cuplaHelper/ValidateCall.hpp"
#include "pmacc/memory/Align.hpp"
#include "pmacc/memory/Delete.hpp"

#include <boost/typeof/std/utility.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/filesystem.hpp>


namespace pmacc
{

namespace bmpl = boost::mpl;
namespace bfs = boost::filesystem;

} //namespace pmacc
