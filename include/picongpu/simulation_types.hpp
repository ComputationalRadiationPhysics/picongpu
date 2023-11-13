/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include <pmacc/algorithms/PromoteType.hpp>
#include <pmacc/algorithms/TypeCast.hpp>
#include <pmacc/algorithms/math.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetStringProperties.hpp>
#include <pmacc/traits/NumberOfExchanges.hpp>

#include <cupla/device/math.hpp>

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
    /** g++ 9 creates compile issues when pulling definitions into picongpu namepsace via 'using namespace
     * pmacc::algorithms::precisionCast;' therefore we pull the class and function separate
     */
    using pmacc::algorithms::precisionCast::precisionCast;
    template<typename CastToType, typename Type>
    using TypeCast = pmacc::algorithms::precisionCast::TypeCast<CastToType, Type>;

    using namespace pmacc::algorithms::promoteType;
    using namespace pmacc::traits;

} // namespace picongpu
