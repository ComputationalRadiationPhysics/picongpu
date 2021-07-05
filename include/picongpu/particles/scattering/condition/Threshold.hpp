/* Copyright 2021 Pawel Ordyna
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
#include "picongpu/particles/scattering/generic/FreeRng.hpp"


namespace picongpu
{
    namespace particles
    {
        namespace scattering
        {
            namespace condition
            {
                namespace acc
                {
                    template<typename T_ParamClass>
                    struct Threshold
                    {
                        using ParamClass = T_ParamClass;
                        static constexpr float_X threshold = ParamClass::threshold;

                        template<typename T_Particle>
                        DINLINE bool operator()(T_Particle const& particle, float_X const& density)
                            const
                        {
                            return (density > threshold);
                        }
                    };
                } // namespace acc
            }
        } // namespace scattering
    } // namespace particles
} // namespace picongpu
