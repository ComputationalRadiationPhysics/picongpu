/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
            HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
            {
                if(static_cast<uint32_t>(totalCellOffset.y()) < ParamClass::vacuumCellsY)
                {
                    return 0._X;
                }

                floatD_X const globalCellPos(
                    precisionCast<float_X>(totalCellOffset) * sim.pic.getCellSize().shrink<simDim>());

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
