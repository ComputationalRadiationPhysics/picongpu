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

namespace picongpu::particles::collision::detail
{
    template<
        typename T_FramePtr,
        typename T_Worker,
        typename T_ForEachCell,
        typename T_EntryListArray,
        typename T_Array,
        typename T_Filter>
    DINLINE void cellDensity(
        T_Worker const& worker,
        T_ForEachCell forEachCell,
        T_EntryListArray& parCellList,
        T_Array& densityArray,
        T_Filter& filter)
    {
        forEachCell(
            [&](uint32_t const linearIdx)
            {
                auto parAccess = parCellList.template getParticlesAccessor<T_FramePtr>(linearIdx);
                uint32_t const numParInCell = parAccess.size();
                float_X density(0.0);
                for(uint32_t partIdx = 0; partIdx < numParInCell; partIdx++)
                {
                    auto particle = parAccess[partIdx];
                    density += particle[weighting_];
                }
                densityArray[linearIdx] = density / CELL_VOLUME;
            });
    }
} // namespace picongpu::particles::collision::detail
