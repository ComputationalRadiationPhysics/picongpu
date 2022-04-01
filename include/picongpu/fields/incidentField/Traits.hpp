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
            // Default implementation of the trait declared in profiles/Free.def
            template<typename T_Profile, uint32_t T_axis, int32_t T_direction>
            struct GetFunctorIncidentE
            {
                using type = typename T_Profile::FunctorIncidentE;
            };

            // Default implementation of the trait declared in profiles/Free.def
            template<typename T_Profile, uint32_t T_axis, int32_t T_direction>
            struct GetFunctorIncidentB
            {
                using type = typename T_Profile::FunctorIncidentB;
            };

            /** Type of incident E functor for the given profile type
             *
             * @tparam T_Profile profile type
             * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
             * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the max
             * boundary inwards)
             */
            template<typename T_Profile, uint32_t T_axis, int32_t T_direction>
            using FunctorIncidentE = typename GetFunctorIncidentE<T_Profile, T_axis, T_direction>::type;

            /** Type of incident B functor for the given profile type
             *
             * @tparam T_Profile profile type
             * @tparam T_axis boundary axis, 0 = x, 1 = y, 2 = z
             * @tparam T_direction direction, 1 = positive (from the min boundary inwards), -1 = negative (from the max
             * boundary inwards)
             */
            template<typename T_Profile, uint32_t T_axis, int32_t T_direction>
            using FunctorIncidentB = typename GetFunctorIncidentB<T_Profile, T_axis, T_direction>::type;

            namespace detail
            {
                /** Get ZMin incident field profile
                 *
                 * The resulting field profile is set as ::type.
                 * Implementation for 3d returns ZMin from incidentField.param.
                 *
                 * @tparam T_is3d if the simulation is 3d or not
                 */
                template<bool T_is3d = true>
                struct GetZMin
                {
                    using type = ZMin;
                };

                //! Get None incident field profile type as ZMin in the non-3d case
                template<>
                struct GetZMin<false>
                {
                    using type = profiles::None;
                };

                /** Get ZMax incident field profile type
                 *
                 * The resulting field profile is set as ::type.
                 * Implementation for 3d returns ZMax from incidentField.param.
                 *
                 * @tparam T_is3d is the simulation 3d
                 */
                template<bool T_is3d = true>
                struct GetZMax
                {
                    using type = ZMax;
                };

                //! Get None incident field profile type as ZMax in the non-3d case
                template<>
                struct GetZMax<false>
                {
                    using type = profiles::None;
                };

                //! ZMin incident field profile type alias adjusted for dimensionality
                using ZMin = GetZMin<simDim == 3>::type;

                //! ZMax incident field profile type alias adjusted for dimensionality
                using ZMax = GetZMax<simDim == 3>::type;

            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
