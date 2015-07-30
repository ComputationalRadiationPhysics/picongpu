/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include "simulation_defines.hpp"
#include "simulationControl/MovingWindow.hpp"

namespace picongpu
{

namespace gasProfiles
{

template<typename T_ParamClass>
struct GaussianCloudImpl : public T_ParamClass
{
    typedef T_ParamClass ParamClass;

    template<typename T_SpeciesType>
    struct apply
    {
        typedef GaussianCloudImpl<ParamClass> type;
    };

    HINLINE GaussianCloudImpl(uint32_t currentStep)
    {
    }

    /** Calculate the gas density
     *
     * @param totalCellOffset total offset including all slides [in cells]
     */
    HDINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
    {
        const float_64 unit_length = UNIT_LENGTH;
        const float_X vacuum_y = float_X(ParamClass::vacuumCellsY) * cellSize.y();
        const floatD_X center = precisionCast<float_32>(ParamClass::center_SI / unit_length);
        const floatD_X sigma = precisionCast<float_32>(ParamClass::sigma_SI / unit_length);


        const floatD_X globalCellPos(
                                     precisionCast<float_X>(totalCellOffset) *
                                     cellSize
                                     );

        if (globalCellPos.y() < vacuum_y) return float_X(0.0);

        float_X density(1.0);
        const float_X power = ParamClass::gasPower;
        for (uint32_t d = 0; d < simDim; ++d)
        {
            const float_X exponent(math::abs((globalCellPos[d] - center[d]) / sigma[d]));
            density *= math::exp(ParamClass::gasFactor * math::pow(exponent, power));
        }

        return density;
    }

};
} //namespace gasProfiles
} //namespace picongpu
