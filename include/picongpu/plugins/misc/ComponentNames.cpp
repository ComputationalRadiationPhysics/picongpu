/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
