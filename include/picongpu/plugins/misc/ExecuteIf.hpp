/*
 * SPDX-FileCopyrightText: Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/param.hpp"

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
