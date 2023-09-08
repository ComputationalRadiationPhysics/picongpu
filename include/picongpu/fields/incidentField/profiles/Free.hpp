/* Copyright 2022-2023 Sergei Bastrakov
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

#include "picongpu/fields/incidentField/Traits.hpp"
#include "picongpu/fields/incidentField/profiles/Free.def"

#include <cstdint>
#include <string>


namespace picongpu
{
    namespace fields
    {
        namespace incidentField
        {
            namespace profiles
            {
                template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
                struct Free
                {
                    //! Get text name of the incident field profile
                    HINLINE static std::string getName()
                    {
                        return "Free";
                    }
                };
            } // namespace profiles

            namespace detail
            {
                /** Get type of incident field E functor for the free profile type
                 *
                 * @tparam T_FunctorIncidentE functor for the incident E field
                 * @tparam T_FunctorIncidentB functor for the incident B field
                 */
                template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
                struct GetFunctorIncidentE<profiles::Free<T_FunctorIncidentE, T_FunctorIncidentB>>
                {
                    using type = T_FunctorIncidentE;
                };

                /** Get type of incident field B functor for the free profile type
                 *
                 * @tparam T_FunctorIncidentE functor for the incident E field
                 * @tparam T_FunctorIncidentB functor for the incident B field
                 */
                template<typename T_FunctorIncidentE, typename T_FunctorIncidentB>
                struct GetFunctorIncidentB<profiles::Free<T_FunctorIncidentE, T_FunctorIncidentB>>
                {
                    using type = T_FunctorIncidentB;
                };

            } // namespace detail
        } // namespace incidentField
    } // namespace fields
} // namespace picongpu
