/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <numeric>
#include <string>

namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** concatenate all values of an string container
             *
             * @tparam T_Container type of the container
             *
             * @param vector source container (required interface: `begin(), end()`)
             * @param separator separator between two elements
             */
            template<typename T_Container>
            std::string concatenateToString(T_Container& container, std::string const& separator = ",")
            {
                return std::accumulate(
                    container.begin(),
                    container.end(),
                    std::string(),
                    [&](std::string const& result, std::string const& inString)
                    { return result.empty() ? inString : result + separator + inString; });
            }
        } // namespace misc
    } // namespace plugins
} // namespace picongpu
