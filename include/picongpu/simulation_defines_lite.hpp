/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera, Sergei Bastrakov
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "pmacc_renamings.hpp"

#include <pmacc/algorithms/PromoteType.hpp>
#include <pmacc/algorithms/TypeCast.hpp>
#include <pmacc/algorithms/math.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetStringProperties.hpp>
#include <pmacc/traits/NumberOfExchanges.hpp>
#include <pmacc/types.hpp>

#include <cupla/device/math.hpp>

#include <cstdint>

namespace picongpu
{
    using namespace pmacc;
}

namespace picongpu
{
    namespace precision32Bit
    {
        using precisionType = float;
    }

    namespace precision64Bit
    {
        using precisionType = double;
    }

    namespace math = cupla::device::math;
    using namespace pmacc::algorithms::precisionCast;
    using namespace pmacc::algorithms::promoteType;
    using namespace pmacc::traits;
    using namespace picongpu::traits;

} // namespace picongpu

// Manually chosen most important includes.
// Those do not pull any other picongpu includes

// simDim value
#include <picongpu/param/dimension.param>

// float_X and related types, is used also by memory.param
// it also includes precision.unitless inside (but this is no problem)
#include <picongpu/param/precision.param>

// MappingDesc type
#include <picongpu/param/memory.param>
