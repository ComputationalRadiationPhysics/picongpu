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
#include "picongpu/simulation/control/MovingWindow.hpp"


namespace picongpu
{
    namespace densityProfiles
    {
        template<typename T_ParamClass>
        struct GaussianImpl : public T_ParamClass
        {
            using ParamClass = T_ParamClass;

            template<typename T_SpeciesType>
            struct apply
            {
                using type = GaussianImpl<ParamClass>;
            };

            HINLINE GaussianImpl(uint32_t currentStep)
            {
            }

            /** Calculate the normalized density
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
            {
                const float_X vacuum_y = float_X(ParamClass::vacuumCellsY) * cellSize.y();
                const float_X gas_center_left = ParamClass::gasCenterLeft_SI / UNIT_LENGTH;
                const float_X gas_center_right = ParamClass::gasCenterRight_SI / UNIT_LENGTH;
                const float_X gas_sigma_left = ParamClass::gasSigmaLeft_SI / UNIT_LENGTH;
                const float_X gas_sigma_right = ParamClass::gasSigmaRight_SI / UNIT_LENGTH;

                const floatD_X globalCellPos(precisionCast<float_X>(totalCellOffset) * cellSize.shrink<simDim>());

                if(globalCellPos.y() * cellSize.y() < vacuum_y)
                {
                    return float_X(0.0);
                }

                float_X exponent = float_X(0.0);
                if(globalCellPos.y() < gas_center_left)
                {
                    exponent = math::abs((globalCellPos.y() - gas_center_left) / gas_sigma_left);
                }
                else if(globalCellPos.y() >= gas_center_right)
                {
                    exponent = math::abs((globalCellPos.y() - gas_center_right) / gas_sigma_right);
                }

                const float_X gas_power = ParamClass::gasPower;
                const float_X density = math::exp(float_X(ParamClass::gasFactor) * math::pow(exponent, gas_power));
                return density;
            }
        };
    } // namespace densityProfiles
} // namespace picongpu
