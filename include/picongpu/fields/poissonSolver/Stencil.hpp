/* Copyright 2025 Tapish Narwal, Luca Pennati, Rene Widera
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
#include "picongpu/fields/FieldTmpOperations.hpp"
#include "picongpu/fields/poissonSolver/FieldV.hpp"

#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/ExchangeMapping.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>

namespace picongpu::fields::poissonSolver
{
    struct StencilFunc
    {
        HDINLINE auto operator()(auto fieldIn) const
        {
            constexpr auto cellSize = sim.pic.getCellSize().shrink<simDim>();
            constexpr auto cellSizeSquared = cellSize * cellSize;
            float_64 const fc0 = 2.0 * (1.0 / (cellSizeSquared)).sumOfComponents();
            if constexpr(simDim == 3u)
            {
                constexpr DataSpace<3u> xDir(1, 0, 0);
                constexpr DataSpace<3u> yDir(0, 1, 0);
                constexpr DataSpace<3u> zDir(0, 0, 1);


                return *fieldIn * fc0 - (fieldIn(-xDir) + fieldIn(xDir)) / cellSizeSquared.x()
                       - (fieldIn(-yDir) + fieldIn(yDir)) / cellSizeSquared.y()
                       - (fieldIn(-zDir) + fieldIn(zDir)) / cellSizeSquared.z();
            }
            else if constexpr(simDim == 2u)
            {
                constexpr DataSpace<2u> xDir(1, 0);
                constexpr DataSpace<2u> yDir(0, 1);

                return *fieldIn * fc0 - (fieldIn(-xDir) + fieldIn(xDir)) / cellSizeSquared.x()
                       - (fieldIn(-yDir) + fieldIn(yDir)) / cellSizeSquared.y();
            }
        }
    };

    struct Stencil
    {
        DINLINE auto operator()(
            auto const& worker,
            auto const mapper,
            auto const stencilFunctor,
            auto fieldOut,
            auto fieldIn) const -> void
        {
            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
            DataSpace<simDim> superCellCellOffset = superCellIdx * SuperCellSize::toRT();

            using Type = typename decltype(fieldIn)::ValueType;
            using BlockArea = pmacc::SuperCellDescription<
                SuperCellSize,
                typename pmacc::math::CT::make_Int<SuperCellSize::dim, 1>::type,
                typename pmacc::math::CT::make_Int<SuperCellSize::dim, 1>::type>;

            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            // use the cached buffer, beacuse I am doing multiple reads, moves the blockArea to shared memory
            auto cache = pmacc::CachedBox::create<0, Type>(worker, BlockArea());
            auto buffShifted = fieldIn.shift(superCellCellOffset);

            // the thread collective is a convenience wrapper for lockstep make for each
            // it deals with the guard offset, subtracts the origin offset
            auto collective = pmacc::makeThreadCollective<BlockArea>();

            pmacc::math::operation::Assign assign;
            collective(worker, assign, cache, buffShifted);

            worker.sync();

            auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

            forEachCellInSupercell(
                [&](int32_t const linearCellIdx)
                {
                    /* cell index within the superCell */
                    DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                    fieldOut[superCellCellOffset + cellIdx] = stencilFunctor(cache.shift(cellIdx));
                });
        }
    };
} // namespace picongpu::fields::poissonSolver
