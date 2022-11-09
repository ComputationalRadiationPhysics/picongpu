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
    namespace particles
    {
        namespace externalBeam
        {
            namespace beam
            {
                namespace beamShapes
                {
                    template<typename T_ParamClass>
                    struct ConstShape
                    {
                        static constexpr float_X startTime = T_ParamClass::startTime_SI / UNIT_TIME;
                        static constexpr float_X endTime = T_ParamClass::endTime_SI / UNIT_TIME;
                        static HDINLINE constexpr float_X getFactor(const float_X& time)
                        {
                            if((time >= startTime || !T_ParamClass::limitStart)
                               && (time <= endTime || !T_ParamClass::limitEnd))
                            {
                                return 1.0_X;
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
    } // namespace particles
} // namespace picongpu
