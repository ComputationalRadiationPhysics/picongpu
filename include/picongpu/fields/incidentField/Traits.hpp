/* Copyright 2020-2022 Sergei Bastrakov
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

#include "picongpu/fields/incidentField/profiles/profiles.hpp"

#include <cstdint>

namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace detail
            {
                /** Get type of incident field functor for the given profile type, axis and direction
                 *
                 * The resulting functor is set as ::type.
                 * By default forwards internal type FunctorIncidentE/FunctorIncidentB.
                 *
                 * These traits have to be specialized by all non-trivial profiles.
                 *
                 * @tparam T_Profile profile type
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 *
                 * @{
                 */

                //! Get functor for incident E values
                template<typename T_Profile, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentE
                {
                    using type = typename T_Profile::FunctorIncidentE;
                };

                //! Get functor for incident B values
                template<typename T_Profile, uint32_t T_axis, int32_t T_direction>
                struct GetFunctorIncidentB
                {
                    using type = typename T_Profile::FunctorIncidentB;
                };

                /** @} */

                /** Type of incident E/B functor for the given profile type
                 *
                 * These are helper aliases to wrap GetFunctorIncidentE/B.
                 * The latter present customization points.
                 *
                 * @tparam T_Profile profile type
                 * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
                 * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the
                 * max boundary inwards)
                 *
                 * @{
                 */

                //! Functor for incident E values
                template<typename T_Profile, uint32_t T_axis, int32_t T_direction>
                using FunctorIncidentE = typename GetFunctorIncidentE<T_Profile, T_axis, T_direction>::type;

                //! Functor for incident B values
                template<typename T_Profile, uint32_t T_axis, int32_t T_direction>
                using FunctorIncidentB = typename GetFunctorIncidentB<T_Profile, T_axis, T_direction>::type;

                /** @} */

            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
