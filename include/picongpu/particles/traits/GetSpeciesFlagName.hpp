/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/HasFlag.hpp>
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
                typename pmacc::traits::GetFlagType<typename T_Species::FrameType, T_Flag>::type>::type;

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
