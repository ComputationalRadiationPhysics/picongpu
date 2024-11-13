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

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/MinimumAndMaximumEFieldNormOfSuperCell.hpp"

#include <pmacc/lockstep/ForEach.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics
{
    struct MinimumAndMaximumEFieldNormOfSuperCell
    {
        using VectorIdx = DataSpace<picongpu::simDim>;

        template<typename T_Worker, typename T_EFieldDataBox>
        HDINLINE static void find(
            T_Worker const& worker,
            VectorIdx const superCellIdx,
            T_EFieldDataBox const eFieldBox,
            float_X& minEFieldNormSuperCell,
            float_X& maxEFieldNormSuperCell)
        {
            auto initAccumulationVariables = lockstep::makeMaster(worker);
            initAccumulationVariables(
                [&minEFieldNormSuperCell, &maxEFieldNormSuperCell]()
                {
                    /// needs to be initialized with neutral element of Minimum
                    /// @warning never increase the result from this variable, may be maximum representable value.
                    minEFieldNormSuperCell = std::numeric_limits<float_X>::max();
                    maxEFieldNormSuperCell = 0._X;
                });
            worker.sync();

            constexpr auto numberCellsInSuperCell
                = pmacc::math::CT::volume<typename picongpu::SuperCellSize>::type::value;
            VectorIdx const superCellCellOffset = superCellIdx * picongpu::SuperCellSize::toRT();
            auto forEachCell = pmacc::lockstep::makeForEach<numberCellsInSuperCell>(worker);

            /// @todo switch to shared memory reduce, Brian Marre, 2024
            forEachCell(
                [&worker, &superCellCellOffset, &maxEFieldNormSuperCell, &minEFieldNormSuperCell, &eFieldBox](
                    uint32_t const linearCellIdx)
                {
                    VectorIdx const localCellIndex
                        = pmacc::math::mapToND(picongpu::SuperCellSize::toRT(), static_cast<int>(linearCellIdx));
                    VectorIdx const cellIndex = localCellIndex + superCellCellOffset;

                    auto const eFieldNorm = pmacc::math::l2norm(eFieldBox(cellIndex));

                    alpaka::atomicMin(
                        worker.getAcc(),
                        // unit: unit_eField
                        &minEFieldNormSuperCell,
                        eFieldNorm);

                    alpaka::atomicMax(
                        worker.getAcc(),
                        // unit: unit_eField
                        &maxEFieldNormSuperCell,
                        eFieldNorm);
                });
        }
    };
} // namespace picongpu::particles::atomicPhysics
