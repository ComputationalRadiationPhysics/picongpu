/* Copyright 2016-2021 Axel Huebl
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

#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <string>


namespace picongpu
{
    namespace traits
    {
        /** Get the GetStringProperties "name" attribute of a Species' Flag
         *
         * Returns the "name" attribute of a species string attribute list as
         * std::string and if not present, returns "none".
         */
        template<
            typename T_Species,
            typename T_Flag,
            bool T_hasFlag = HasFlag<typename T_Species::FrameType, T_Flag>::type::value>
        struct GetSpeciesFlagName
        {
            using SpeciesFlag = typename pmacc::traits::Resolve<
                typename GetFlagType<typename T_Species::FrameType, T_Flag>::type>::type;

            std::string operator()() const
            {
                GetStringProperties<SpeciesFlag> stringProps;
                return stringProps["name"].value;
            }
        };

        template<typename T_Species, typename T_Flag>
        struct GetSpeciesFlagName<T_Species, T_Flag, false>
        {
            std::string operator()() const
            {
                return "none";
            }
        };
    } // namespace traits
} // namespace picongpu
