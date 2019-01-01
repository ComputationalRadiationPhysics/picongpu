/* Copyright 2015-2019 Rene Widera, Richard Pausch, Axel Huebl
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
#include "picongpu/simulationControl/MovingWindow.hpp"


namespace picongpu
{
namespace densityProfiles
{

template<typename T_ParamClass>
struct FreeFormulaImpl : public T_ParamClass
{
    using ParamClass = T_ParamClass;

    template<typename T_SpeciesType>
    struct apply
    {
        using type = FreeFormulaImpl<ParamClass>;
    };

    HINLINE FreeFormulaImpl(uint32_t currentStep)
    {
    }

    /** Calculate the normalized density
     *
     * @param totalCellOffset total offset including all slides [in cells]
     */
    HDINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
    {
        const float_64 unitLength(UNIT_LENGTH); // workaround to use UNIT_LENGTH on device
        const float3_64 cellSize_SI( precisionCast<float_64>(cellSize) * unitLength );
        const floatD_64 position_SI( precisionCast<float_64>(totalCellOffset) * cellSize_SI.shrink<simDim>( ) );

        return ParamClass::operator()(position_SI, cellSize_SI);
    }
};
}
}
