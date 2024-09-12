/* Copyright 2020-2023 Sergei Bastrakov, Finn-Ole Carstens
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

namespace picongpu::fields::incidentField
{
    //! Helper incident field functor always returning 0
    struct ZeroFunctor
    {
        /** Create a functor on the host side for the given time step
         *
         * @param currentStep current time step index, note that it is fractional
         * @param unitField conversion factor from SI to internal units,
         *                  field_internal = field_SI / unitField
         */
        HINLINE ZeroFunctor(float_X const currentStep, float3_64 const unitField)
        {
        }

        /** Return zero incident field for any given position
         *
         * @param totalCellIdx cell index in the total domain (including all moving window slides),
         *        note that it is fractional
         * @return incident field value in internal units
         */
        HDINLINE float3_X operator()(floatD_X const& totalCellIdx) const
        {
            return float3_X::create(0.0_X);
        }
    };
} // namespace picongpu::fields::incidentField
