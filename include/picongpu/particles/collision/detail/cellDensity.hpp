/* Copyright 2020-2023 Pawel Ordyna
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

#include "picongpu/particles/collision/detail/ListEntry.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            namespace detail
            {
                template<
                    typename T_Worker,
                    typename T_ForEach,
                    typename T_FramePtr,
                    typename T_ParBox,
                    typename T_EntryListArray,
                    typename T_Array,
                    typename T_Filter>
                DINLINE void cellDensity(
                    T_Worker const& worker,
                    T_ForEach forEach,
                    T_FramePtr firstFrame,
                    T_ParBox parBox,
                    T_EntryListArray& parCellList,
                    T_Array& densityArray,
                    T_Filter& filter)
                {
                    forEach(
                        [&](uint32_t const linearIdx)
                        {
                            uint32_t const numParInCell = parCellList[linearIdx].size;
                            uint32_t* parListStart = parCellList[linearIdx].ptrToIndicies;
                            float_X density(0.0);
                            for(uint32_t ii = 0; ii < numParInCell; ii++)
                            {
                                auto particle = getParticle(parBox, firstFrame, parListStart[ii]);
                                if(filter(worker, particle))
                                {
                                    density += particle[weighting_];
                                }
                            }
                            densityArray[linearIdx] = density / CELL_VOLUME;
                        });
                }

            } // namespace detail
        } // namespace collision
    } // namespace particles
} // namespace picongpu
