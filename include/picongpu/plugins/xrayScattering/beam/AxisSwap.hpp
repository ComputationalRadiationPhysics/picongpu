/* Copyright 2020-2021 Pawel Ordyna
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
    namespace plugins
    {
        namespace xrayScattering
        {
            namespace beam
            {
                /** Swaps vector axes and multiplies the result with an integer vector.
                 *
                 * When the integer vector contains only 1 and -1 values, this swap
                 * correspond to a vector rotation that consists only out of right angle
                 * subrotations.
                 *
                 * @tparam axis0 Which old axis (0,1 or 2) is the new first axis (0).
                 * @tparam axis1 Which old axis (0,1 or 2) is the new second axis (1).
                 * @tparam axis2 Which old axis (0,1 or 2) is the new third axis (2).
                 * @tparam a0 Integer vector first component.
                 * @tparam a1 Integer vector second component.
                 * @tparam a2 Integer vector third component.
                 */
                template<unsigned axis0, unsigned axis1, unsigned axis2, int a0, int a1, int a2>
                struct AxisSwap
                {
                    //! Performs the axis swap and the multiplication.
                    static HDINLINE float3_X rotate(float3_X const& vec)
                    {
                        return float3_X(a0 * vec[axis0], a1 * vec[axis1], a2 * vec[axis2]);
                    }

                    //! Performs the reversed operation (back rotation).
                    static HDINLINE float3_X reverse(float3_X const& vec)
                    {
                        PMACC_ASSERT(a0 != 0);
                        PMACC_ASSERT(a1 != 0);
                        PMACC_ASSERT(a2 != 0);

                        float3_X result;
                        result[axis0] = vec[0] / a0;
                        result[axis1] = vec[1] / a1;
                        result[axis2] = vec[2] / a2;
                        return result;
                    }
                };
            } // namespace beam
        } // namespace xrayScattering
    } // namespace plugins
} // namespace picongpu
