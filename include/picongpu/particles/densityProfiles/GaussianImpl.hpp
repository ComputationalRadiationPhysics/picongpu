/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"
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
                floatD_X const globalCellPos(
                    precisionCast<float_X>(totalCellOffset) * sim.pic.getCellSize().shrink<simDim>());

                float_X const vacuum_y = float_X(ParamClass::vacuumCellsY) * sim.pic.getCellSize().y();
                if(globalCellPos.y() * sim.pic.getCellSize().y() < vacuum_y)
                {
                    return 0._X;
                }

                constexpr float_X gasCenterLeft
                    = static_cast<float_X>(ParamClass::gasCenterLeft_SI / sim.unit.length());
                constexpr float_X gasCenterRight
                    = static_cast<float_X>(ParamClass::gasCenterRight_SI / sim.unit.length());
                constexpr float_X gasSigmaLeft = static_cast<float_X>(ParamClass::gasSigmaLeft_SI / sim.unit.length());
                constexpr float_X gasSigmaRight
                    = static_cast<float_X>(ParamClass::gasSigmaRight_SI / sim.unit.length());

                auto exponent = 0._X;
                if(globalCellPos.y() < gasCenterLeft)
                {
                    exponent = math::abs((globalCellPos.y() - gasCenterLeft) / gasSigmaLeft);
                }
                else if(globalCellPos.y() >= gasCenterRight)
                {
                    exponent = math::abs((globalCellPos.y() - gasCenterRight) / gasSigmaRight);
                }

                constexpr float_X gasPower = ParamClass::gasPower;
                constexpr float_X gasFactor = ParamClass::gasFactor;
                constexpr float_X densityFunctor = ParamClass::densityFactor;

                float_X const density = densityFunctor * math::exp(gasFactor * math::pow(exponent, gasPower));
                return density;
            }
        };
    } // namespace densityProfiles
} // namespace picongpu
