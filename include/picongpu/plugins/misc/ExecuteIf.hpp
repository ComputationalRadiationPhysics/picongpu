/* Copyright 2024 Tapish Narwal
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

#include "picongpu/simulation_defines.hpp"

#include <pmacc/meta/errorHandlerPolicies/ReturnType.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <string>
#include <type_traits>

namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /**
             * Predicate which checks if string argument is same as compile time species name
             *
             * @tparam T_Species The PMACC cstring or type of the species
             * @param s String holding the species name
             */
            template<typename T_Species>
            struct SpeciesNameIsEqual
            {
                using Species = pmacc::particles::meta::
                    FindByNameOrType_t<VectorAllSpecies, T_Species, pmacc::errorHandlerPolicies::ReturnType<void>>;

                bool operator()(std::string const& s) const
                {
                    if constexpr(std::is_same_v<void, Species>)
                        return false;
                    else
                        return s == Species::FrameType::getName();
                }
            };

            struct ExecuteIf
            {
                /**
                 * Conditionally execute a nullary functor
                 *
                 * @param functor A nullary callable
                 * @param predicate The predicate that determines whether to execute the functor
                 * @param args Variable number of arguments taken by the predicate
                 */
                template<typename T_Callable, typename T_Predicate, typename... T_Args>
                void operator()(T_Callable const& functor, T_Predicate const& predicate, T_Args const&... args) const
                {
                    if(predicate(args...))
                        functor();
                }
            };

        } // namespace misc
    } // namespace plugins
} // namespace picongpu
