/* Copyright 2023 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/device/threadInfo.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/operation.hpp>

struct SetBoundaryConditions
{
    /** initialize each cell
     *
     * @tparam T_BoxWriteOnly PMacc::DataBox, box type of the new grid data
     * @tparam T_Mapping mapping functor type
     *
     * @param buffRead buffer with cell data of the current step
     * @param mapper functor to map a block to a supercell
     */
    template<typename T_Box, typename T_Mapping, typename T_Worker>
    DINLINE void operator()(
        T_Worker const& worker,
        T_Box buff1,
        uint32_t const devicesPerDim,
        pmacc::DataSpace<DIM2> const& globalPos,
        pmacc::DataSpace<DIM2> const& localOffset,
        pmacc::DataSpace<DIM2> const& gridSize,
        T_Mapping const& mapper) const
    {
        // check if gpu is a border gpu
        bool xBorder = globalPos.x() % devicesPerDim == 0 || globalPos.x() % devicesPerDim == devicesPerDim - 1;
        bool yBorder = globalPos.y() % devicesPerDim == 0 || globalPos.y() % devicesPerDim == devicesPerDim - 1;
        // return immidiately if not a border gpu
        if(!(xBorder || yBorder))
        {
            return;
        }

        using SuperCellSize = typename T_Mapping::SuperCellSize;
        constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

        // get position in grid in units of SuperCells from blockID
        pmacc::DataSpace<DIM2> const block(mapper.getSuperCellIndex(pmacc::DataSpace<DIM2>(worker.blockDomIdxND())));
        // convert position in unit of cells
        pmacc::DataSpace<DIM2> const blockCell = block * SuperCellSize::toRT();

        pmacc::lockstep::makeForEach<cellsPerSuperCell>(worker)(
            [&](int32_t const linearIdx)
            {
                pmacc::DataSpace<DIM2> bufPos = blockCell + pmacc::math::mapToND(SuperCellSize::toRT(), linearIdx);

                pmacc::DataSpace<DIM2> pos = localOffset + bufPos - SuperCellSize::toRT();

                // global border means pos < 0 or > grid size
                if(pos.x() == 0 || pos.x() == gridSize.x() - 1 || pos.y() == 0 || pos.y() == gridSize.y() - 1)
                {
                    // Set all borders as hot == 1.
                    buff1(bufPos) = 1.;
                }
            });
    };
};
