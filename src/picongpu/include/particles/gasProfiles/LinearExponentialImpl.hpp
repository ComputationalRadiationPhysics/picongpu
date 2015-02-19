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

namespace picongpu
{

namespace gasProfiles
{

template<typename T_ParamClass>
struct LinearExponentialImpl : public T_ParamClass
{
    typedef T_ParamClass ParamClass;

    template<typename T_SpeciesType>
    struct apply
    {
        typedef LinearExponentialImpl<ParamClass> type;
    };

    HINLINE LinearExponentialImpl(uint32_t currentStep)
    {

    }

    /* Calculate the gas density
     *
     * @param totalCellOffset total offset including all slides [in cells]
     */
    HDINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
    {
        const float_X vacuum_y = float_64(ParamClass::VACUUM_CELLS_Y) * cellSize.y();
        const float_X gas_a = ParamClass::SI::GAS_A * UNIT_LENGTH;
        const float_X gas_d = ParamClass::SI::GAS_D * UNIT_LENGTH;
        const float_X gas_y_max = ParamClass::SI::GAS_Y_MAX / UNIT_LENGTH;

        const floatD_X globalCellPos(
                                     precisionCast<float_X>(totalCellOffset) *
                                     cellSize
                                     );
        float_X density = float_X(0.0);

        if (globalCellPos.y() < vacuum_y) return density;

        if (globalCellPos.y() <= gas_y_max) // linear slope
            density = gas_a * globalCellPos.y() + ParamClass::GAS_B;
        else // exponential slope
            density = math::exp((globalCellPos.y() - gas_y_max) * gas_d);

        // avoid < 0 densities for the linear slope
        if (density < float_X(0.0))
            density = float_X(0.0);

        return density;
    }
};
} //namespace gasProfiles
} //namespace picongpu
