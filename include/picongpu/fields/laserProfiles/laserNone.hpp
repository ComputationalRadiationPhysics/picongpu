/* Copyright 2013-2018 Axel Huebl, Heiko Burau, Rene Widera
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

#include <pmacc/types.hpp>
#include "picongpu/simulation_defines.hpp"

namespace picongpu
{
    /** No Laser initialization
     *
     */
    namespace laserNone
    {

        /** Compute the
         *
         */
        HINLINE float3_X laserLongitudinal( uint32_t, float_X& phase )
        {
            const float3_X elong(float3_X::create(0.0));

            phase = float_X(0.0);

            return elong;
        }

        /**
         *
         * @param elong
         * @param phase
         * @param posX
         * @param posZ
         * @return
         */
        HDINLINE float3_X laserTransversal( float3_X elong, const float_X, const float_X, const float_X )
        {
            return elong;
        }

    }
}

