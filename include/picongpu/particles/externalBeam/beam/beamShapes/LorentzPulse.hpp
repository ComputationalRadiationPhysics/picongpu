/* Copyright 2020-2023 Pawel Ordyna
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
    namespace paticles
    {
        namespace externalBeam
        {
            namespace beam
            {
                namespace beamShapes
                {
                    template<typename T_ParamClass>
                    struct LorentzPulse
                    {
                        static constexpr float_X timeOffset = T_ParamClass::timeOffset_SI / UNIT_TIME;
                        static constexpr float_X cutTimeFront = T_ParamClass::cutTimeFront_SI / UNIT_TIME;
                        static constexpr float_X cutTimeBack = T_ParamClass::cutTimeBack_SI / UNIT_TIME;
                        static constexpr float_X startTime = timeOffset - cutTimeFront;
                        static constexpr float_X endTime = timeOffset + cutTimeBack;

                        static constexpr float_X gamma = T_ParamClass::FWHM_SI / UNIT_TIME / 2.0;
                        static constexpr float_X gammaSquared = gamma * gamma;


                        static HDINLINE constexpr float_X getFactor(const float_X& time)
                        {
                            if((time >= startTime || !T_ParamClass::limitStart)
                               && (time <= endTime || !T_ParamClass::limitEnd))
                            {
                                const float_X timeDist = time - timeOffset;
                                return 1.0_X / (1.0_X + (timeDist * timeDist / gammaSquared));
                            }
                            else
                            {
                                return 0.0_X;
                            }
                        }
                    };
                } // namespace beamShapes
            } // namespace beam
        } // namespace externalBeam
    } // namespace paticles
} // namespace picongpu
