/* Copyright 2017-2021 Rene Widera
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
