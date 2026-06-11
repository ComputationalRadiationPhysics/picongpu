/*
 * SPDX-FileCopyrightText: Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
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
                auto parAccess = parCellList.getParticlesAccessor(linearIdx);
                uint32_t const numParInCell = parAccess.size();
                float_X density(0.0);
                for(uint32_t partIdx = 0; partIdx < numParInCell; partIdx++)
                {
                    auto particle = parAccess[partIdx];
                    density += particle[weighting_];
                }
                densityArray[linearIdx] = density / sim.pic.getCellSize().productOfComponents();
            });
    }
} // namespace picongpu::particles::collision::detail
