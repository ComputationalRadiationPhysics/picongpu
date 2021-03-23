/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include "version.hpp"
#include <pmacc/algorithms/TypeCast.hpp>
#include <pmacc/algorithms/PromoteType.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/algorithms/math.hpp>
#include <cupla/device/math.hpp>
#include <pmacc/traits/GetStringProperties.hpp>
#include "picongpu/traits/GetMargin.hpp"
#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/NumberOfExchanges.hpp>
#include "picongpu/traits/GetDataBoxType.hpp"

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
