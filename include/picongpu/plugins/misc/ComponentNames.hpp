/* Copyright 2020-2021 Sergei Bastrakov
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

#include <string>
#include <vector>


namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** Get text names of vector components
             *
             * For 1-4 and 6 components use predefined names,
             * for other amounts use generic different names
             *
             * @param numComponents number of components
             */
            std::vector<std::string> getComponentNames(uint32_t numComponents);

        } // namespace misc
    } // namespace plugins
} // namespace picongpu
