/* Copyright 2013-2023 Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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

#include <alpaka/alpaka.hpp>

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
#    include <mallocMC/mallocMC.hpp>
#endif


#include <pmacc/boost_workaround.hpp>

#include "pmacc/alpakaHelper/ValidateCall.hpp"
#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/attribute/Constexpr.hpp"
#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/debug/PMaccVerbose.hpp"
#include "pmacc/dimensions/Definition.hpp"
#include "pmacc/eventSystem/EventType.hpp"
#include "pmacc/memory/Align.hpp"
#include "pmacc/meta/Mp11.hpp"
#include "pmacc/ppFunctions.hpp"
#include "pmacc/type/Area.hpp"
#include "pmacc/type/Exchange.hpp"
#include "pmacc/type/Integral.hpp"

#include <alpaka/alpaka.hpp>

#include <boost/filesystem.hpp>


namespace pmacc
{
} // namespace pmacc
