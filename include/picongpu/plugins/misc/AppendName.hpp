/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <string>
#include <vector>

namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** append the name of an filter to a vector
             *
             * @tparam T_Filter filter class (required interface: `getName( )`)
             */
            template<typename T_Filter>
            struct AppendName
            {
                void operator()(std::vector<std::string>& vector) const
                {
                    vector.emplace_back(T_Filter::getName());
                }
            };

            template<typename T_Species>
            struct AppendSpeciesName
            {
                void operator()(std::vector<std::string>& vector) const
                {
                    vector.emplace_back(T_Species::FrameType::getName());
                }
            };

        } // namespace misc
    } // namespace plugins
} // namespace picongpu
