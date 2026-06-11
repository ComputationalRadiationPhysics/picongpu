/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <algorithm>

namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** search for an element within a STL container
             *
             * @tparam T_Container standard container, type of the container
             *
             * @param container object to query
             * @param value object to search
             * @return true if container contains the element, else false
             */
            template<typename T_Container>
            bool containsObject(T_Container const& container, typename T_Container::value_type const& value)
            {
                auto it = std::find(container.begin(), container.end(), value);

                return it != container.end();
            }
        } // namespace misc
    } // namespace plugins
} // namespace picongpu
