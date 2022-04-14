/* Copyright 2022 Sergei Bastrakov
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

#include "picongpu/fields/incidentField/Functors.hpp"
#include "picongpu/fields/incidentField/Traits.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace profiles
            {
                namespace detail
                {
                    /** Base functor for calculating incident E functors
                     *
                     * Defines internal coordinate system tied to given axis and direction.
                     * Checks unit matching.
                     *
                     * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                     * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from
                     * the max boundary inwards)
                     */
                    template<uint32_t T_axis, int32_t T_direction>
                    struct BaseFunctorE
                    {
                    public:
                        //! Propagation axis
                        constexpr static uint32_t axis = T_axis;

                        //! Propagation direction
                        constexpr static uint32_t direction = T_direction;

                        /** Internal right-side coordinate system (dir0, dir1, dir2) with dir0 being propagation axis,
                         * dir1 being orthogonal to dir0 and dir2 = cross(dir0, dir1).
                         * @{
                         */
                        constexpr static uint32_t dir0 = axis;
                        constexpr static uint32_t dir1 = (dir0 + 1) % 3;
                        constexpr static uint32_t dir2 = (dir0 + 2) % 3;
                        /** @} */

                        /** Create a functor on the host side, check that unit matches the internal E unit
                         *
                         * @param unitField conversion factor from SI to internal units,
                         *                  fieldE_internal = fieldE_SI / unitField
                         */
                        HINLINE BaseFunctorE(const float3_64 unitField)
                        {
                            // Ensure that we always get unitField = (UNIT_EFIELD, UNIT_EFIELD, UNIT_EFIELD) so that
                            // we can always calculate in internal units and avoid conversions in child types.
                            // We can afford it each time, as this is done on host before kernel
                            for(uint32_t axis = 0; axis < 3; axis++)
                            {
                                constexpr double ulp = 1.0;
                                constexpr double eps = std::numeric_limits<double>::epsilon();
                                bool const isMatchingUnit = (std::fabs(unitField[axis] - UNIT_EFIELD) <= eps * ulp);
                                if(!isMatchingUnit)
                                    throw std::runtime_error(
                                        "Incident field BaseFunctorE created with wrong unit: expected "
                                        + std::to_string(UNIT_EFIELD) + ", got " + std::to_string(unitField[axis]));
                            }
                        }
                    };
                } // namespace detail
            } // namespace profiles
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
