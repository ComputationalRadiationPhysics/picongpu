/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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


namespace picongpu
{
    namespace densityProfiles
    {
        template<typename T_ParamClass>
        struct LinearExponentialImpl : public T_ParamClass
        {
            using ParamClass = T_ParamClass;

            template<typename T_SpeciesType>
            struct apply
            {
                using type = LinearExponentialImpl<ParamClass>;
            };

            HINLINE LinearExponentialImpl(uint32_t currentStep)
            {
            }

            /* Calculate the normalized density
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
            {
                const float_X vacuum_y = float_X(ParamClass::vacuumCellsY) * cellSize.y();
                const float_X gas_a = ParamClass::gasA_SI * UNIT_LENGTH;
                const float_X gas_d = ParamClass::gasD_SI * UNIT_LENGTH;
                const float_X gas_y_max = ParamClass::gasYMax_SI / UNIT_LENGTH;

                const floatD_X globalCellPos(precisionCast<float_X>(totalCellOffset) * cellSize.shrink<simDim>());
                float_X density = float_X(0.0);

                if(globalCellPos.y() < vacuum_y)
                    return density;

                if(globalCellPos.y() <= gas_y_max) // linear slope
                    density = gas_a * globalCellPos.y() + ParamClass::gasB;
                else // exponential slope
                    density = math::exp((globalCellPos.y() - gas_y_max) * gas_d);

                // avoid < 0 densities for the linear slope
                if(density < float_X(0.0))
                    density = float_X(0.0);

                return density;
            }
        };
    } // namespace densityProfiles
} // namespace picongpu
