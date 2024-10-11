/* Copyright 2024 Brian Marre
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

// need simulation.param for normalisation and units, memory.param for SuperCellSize and dim.param for simDim
#include "picongpu/defines.hpp"

#include <pmacc/lockstep/ForEach.hpp>

#include <limits>

namespace picongpu::particles::atomicPhysics::kernel
{
    struct EFieldNormExtrema
    {
        //! find minimum and maximum value of the e E-Field Norm of the given superCell
        template<typename T_Worker, typename T_EFieldDataBox, typename T_Type>
        HDINLINE static void find(
            T_Worker const& worker,
            pmacc::DataSpace<picongpu::simDim> const superCellIdx,
            T_EFieldDataBox const eFieldBox,
            T_Type& minEFieldSuperCell,
            T_Type& maxEFieldSuperCell)
        {
            auto onlyMaster = lockstep::makeMaster(worker);
            onlyMaster(
                [&maxEFieldSuperCell, &minEFieldSuperCell]()
                {
                    maxEFieldSuperCell = 0._X;
                    minEFieldSuperCell = std::numeric_limits<float_X>::max();
                });
            worker.sync();

            constexpr auto numberCellsInSuperCell
                = pmacc::math::CT::volume<typename picongpu::SuperCellSize>::type::value;
            DataSpace<picongpu::simDim> const superCellCellOffset = superCellIdx * picongpu::SuperCellSize::toRT();
            auto forEachCell = pmacc::lockstep::makeForEach<numberCellsInSuperCell, T_Worker>(worker);

            /// @todo switch to shared memory reduce, Brian Marre, 2024
            forEachCell(
                [&worker, &superCellCellOffset, &maxEFieldSuperCell, &minEFieldSuperCell, &eFieldBox](
                    uint32_t const linearIdx)
                {
                    DataSpace<picongpu::simDim> const localCellIndex
                        = pmacc::math::mapToND(picongpu::SuperCellSize::toRT(), static_cast<int>(linearIdx));
                    DataSpace<picongpu::simDim> const cellIndex = localCellIndex + superCellCellOffset;

                    auto const eFieldNorm = pmacc::math::l2norm(eFieldBox(cellIndex));

                    alpaka::atomicMax(
                        worker.getAcc(),
                        // sim.unit.eField()
                        &maxEFieldSuperCell,
                        eFieldNorm);

                    alpaka::atomicMin(
                        worker.getAcc(),
                        // sim.unit.eField()
                        &minEFieldSuperCell,
                        eFieldNorm);
                });
        }
    };
} // namespace picongpu::particles::atomicPhysics::kernel
