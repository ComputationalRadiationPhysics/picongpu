/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/algorithms/Gamma.def"


namespace picongpu
{
    template<typename T_PrecisionType>
    template<typename T_MomType, typename T_MassType>
    HDINLINE T_PrecisionType Gamma<T_PrecisionType>::operator()(T_MomType const& mom, T_MassType const mass) const
    {
        using namespace pmacc;

        valueType const fMom2 = pmacc::math::abs2(precisionCast<valueType>(mom));
        constexpr valueType c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

        valueType const m2_c2_reci = valueType(1.0) / precisionCast<valueType>(mass * mass * c2);

        return math::sqrt(precisionCast<valueType>(valueType(1.0) + fMom2 * m2_c2_reci));
    }

} // namespace picongpu
