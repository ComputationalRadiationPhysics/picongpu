/* Copyright 2020-2021 Jakob Trojok
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
        namespace ionization
        {
            /** calculates ionization current
             */
            struct JIonizationCalc
            {
                /** Functor calculating ionization current.
                 * Is only called if ionization energy is not zero,
                 * thus we ensure the field is different from zero.
                 */
                HDINLINE float3_X operator()(float_X const ionizationEnergy, float3_X const eField)
                {
                    float3_X jion = ionizationEnergy * eField / pmacc::math::abs2(eField) / DELTA_T / CELL_VOLUME;
                    return jion;
                }
            };
        } // namespace ionization
    } // namespace particles
} // namespace picongpu
