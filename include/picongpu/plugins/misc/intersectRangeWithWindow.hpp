/* Copyright 2017-2023 Rene Widera
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

#include "picongpu/simulation/control/Window.hpp"

#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/pluginSystem/toSlice.hpp>

#include <algorithm>
#include <string>


namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** Calculate window offsets and sizes based on a user given range description.
             *
             * @tparam T_dim number of dimensions the simulation is compiled for
             * @param dataDomain PIConGPU data domain description
             * @param inputWindow Selected window within the data domain e.g. the selected moving window.
             * @param selectedRange String with range description for each dimension. Dimensions are separated by
             *                      comma. The range for a dimension is of the form BEGIN:END where BEGIN is included
             *                      and END excluded from the domain.
             *                      If the begin or end of the range is outside of the data domain the values will be
             *                      cropped to fit the data domain.
             * @return Window description with local and global information.
             */
            template<uint32_t T_dim>
            inline Window intersectRangeWithWindow(
                pmacc::SubGrid<T_dim> const& dataDomain,
                Window const& inputWindow,
                std::string const& selectedRange)
            {
                const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

                auto parsedSlice = pmacc::pluginSystem::toRangeSlice(selectedRange);

                Window resultWindow = inputWindow;
                // Parse the ranges only for dimensions the user is providing.
                for(uint32_t d = 0; d < parsedSlice.size() && d < simDim; ++d)
                {
                    auto range = parsedSlice[d];
                    uint32_t const windowOffset = inputWindow.globalDimensions.offset[d];
                    uint32_t const windowSize = inputWindow.globalDimensions.size[d];
                    uint32_t const windowEnd = windowOffset + windowSize;
                    uint32_t const rangeBegin = range.values[0];
                    // Crop the user given end range to the end of the input window.
                    uint32_t const rangeEnd = std::min(range.values[1], windowEnd);

                    // Crop resulting window begin to the end of the input window.
                    uint32_t const newWindowBegin = std::min(rangeBegin + windowOffset, windowEnd);
                    // Crop resulting window end to the end of the input window.
                    uint32_t const newWindowEnd = std::min(rangeEnd + windowOffset, windowEnd);
                    uint32_t const newWindowSize = newWindowEnd - newWindowBegin;

                    resultWindow.globalDimensions.offset[d] = newWindowBegin;
                    resultWindow.globalDimensions.size[d] = newWindowSize;

                    /* Update offset from the local data domain with respect to the window begin.
                     * In cases where the window begin is inside of the local domain of the device the offset is zero
                     * and not negative.
                     */
                    resultWindow.localDimensions.offset[d]
                        = std::max(0, subGrid.getLocalDomain().offset[d] - static_cast<int>(newWindowBegin));


                    /* If the resulting window is before or after the local data domain the window size will be local
                     * zero because the local domain is outside the window else the local domain is part of the window.
                     */
                    if(static_cast<int>(newWindowEnd) <= subGrid.getLocalDomain().offset[d]
                       || subGrid.getLocalDomain().size[d] + subGrid.getLocalDomain().offset[d]
                           < static_cast<int>(newWindowBegin))
                        resultWindow.localDimensions.size[d] = 0;
                    else
                    {
                        // Calculate which data in the local domain are within the resulting window.
                        auto beginOnLocalDomain
                            = std::max(subGrid.getLocalDomain().offset[d], static_cast<int>(newWindowBegin));
                        auto endOnLocalDomain = std::min(
                            subGrid.getLocalDomain().offset[d] + subGrid.getLocalDomain().size[d],
                            static_cast<int>(newWindowEnd));

                        resultWindow.localDimensions.size[d] = endOnLocalDomain - beginOnLocalDomain;
                    }

                    /* In case no data of the local domain are within the resulting window the local offset must be
                     * zero.
                     * */
                    if(resultWindow.localDimensions.size[d] == 0)
                        resultWindow.localDimensions.offset[d] = 0;
                }
                return resultWindow;
            }
        } // namespace misc
    } // namespace plugins
} // namespace picongpu
