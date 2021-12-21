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

#include "picongpu/plugins/misc/ComponentNames.hpp"

#include <array>
#include <string>
#include <vector>


namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            std::vector<std::string> getComponentNames(uint32_t const numComponents)
            {
                /* For low number of components, fall back to the previously used
                 * "xyzw" naming scheme for backward compatibility
                 */
                if(numComponents <= 4)
                {
                    std::array<std::string, 4> names = {"x", "y", "z", "w"};
                    return std::vector<std::string>{names.begin(), names.begin() + numComponents};
                }
                // Special case for 6 PML components
                else if(numComponents == 6)
                    return {"xy", "xz", "yx", "yz", "zx", "zy"};
                else
                {
                    // Otherwise use different generic names
                    auto result = std::vector<std::string>(numComponents);
                    for(auto i = 0u; i < result.size(); i++)
                        result[i] = "component" + std::to_string(i);
                    return result;
                }
            }

        } // namespace misc
    } // namespace plugins
} // namespace picongpu
