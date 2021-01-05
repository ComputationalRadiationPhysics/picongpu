/* Copyright 2013-2021 Rene Widera, Marco Garten
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "types.hpp"
#include <pmacc/math/Vector.hpp>
#include <pmacc/mappings/threads/ThreadCollective.hpp>
#include <pmacc/nvidia/functors/Assign.hpp>
#include <pmacc/memory/boxes/CachedBox.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>
#include <pmacc/random/distributions/distributions.hpp>
#include <pmacc/random/methods/methods.hpp>
#include <pmacc/random/Random.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/mappings/threads/ForEachIdx.hpp>
#include <pmacc/mappings/threads/IdxConfig.hpp>

#include <memory>

namespace gol
{
    namespace kernel
    {
        using namespace pmacc;

        /** run game of life stencil
         *
         * evaluate each cell in the supercell
         *
         * @tparam T_numWorkers number of workers
         */
        template<uint32_t T_numWorkers>
        struct Evolution
        {
            /** run stencil for a supercell
             *
             * @tparam T_BoxReadOnly PMacc::DataBox, box type of the old grid data
             * @tparam T_BoxWriteOnly PMacc::DataBox, box type of the new grid data
             * @tparam T_Mapping mapping functor type
             *
             * @param buffRead buffer with cell data of the current step
             * @param buffWrite buffer for the updated cell data
             * @param rule description of the rule as bitmap mask
             * @param mapper functor to map a block to a supercell
             */
            template<typename T_BoxReadOnly, typename T_BoxWriteOnly, typename T_Mapping, typename T_Acc>
            DINLINE void operator()(
                T_Acc const& acc,
                T_BoxReadOnly const& buffRead,
                T_BoxWriteOnly& buffWrite,
                uint32_t const rule,
                T_Mapping const& mapper) const
            {
                using namespace mappings::threads;

                using Type = typename T_BoxReadOnly::ValueType;
                using SuperCellSize = typename T_Mapping::SuperCellSize;
                using BlockArea = SuperCellDescription<SuperCellSize, math::CT::Int<1, 1>, math::CT::Int<1, 1>>;
                auto cache = CachedBox::create<0, Type>(acc, BlockArea());

                Space const block(mapper.getSuperCellIndex(Space(cupla::blockIdx(acc))));
                Space const blockCell = block * T_Mapping::SuperCellSize::toRT();

                constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;
                constexpr uint32_t numWorkers = T_numWorkers;
                uint32_t const workerIdx = cupla::threadIdx(acc).x;

                auto buffRead_shifted = buffRead.shift(blockCell);

                ThreadCollective<BlockArea, numWorkers> collective(workerIdx);

                nvidia::functors::Assign assign;
                collective(acc, assign, cache, buffRead_shifted);

                cupla::__syncthreads(acc);

                ForEachIdx<IdxConfig<cellsPerSuperCell, numWorkers>>{
                    workerIdx}([&](uint32_t const linearIdx, uint32_t const) {
                    // cell index within the superCell
                    DataSpace<DIM2> const cellIdx = DataSpaceOperations<DIM2>::template map<SuperCellSize>(linearIdx);

                    Type neighbors = 0;
                    for(uint32_t i = 1; i < 9; ++i)
                    {
                        Space const offset(Mask::getRelativeDirections<DIM2>(i));
                        neighbors += cache(cellIdx + offset);
                    }

                    Type isLife = cache(cellIdx);
                    isLife = static_cast<bool>(((!isLife) * (1 << (neighbors + 9))) & rule)
                        + static_cast<bool>((isLife * (1 << (neighbors))) & rule);

                    buffWrite(blockCell + cellIdx) = isLife;
                });
            }
        };

        /** initialize each cell
         *
         * randomly activate each cell within a supercell
         *
         * @tparam T_numWorkers number of workers
         */
        template<uint32_t T_numWorkers>
        struct RandomInit
        {
            /** initialize each cell
             *
             * @tparam T_BoxWriteOnly PMacc::DataBox, box type of the new grid data
             * @tparam T_Mapping mapping functor type
             *
             * @param buffRead buffer with cell data of the current step
             * @param seed random number generator seed
             * @param threshold threshold to activate a cell, range [0.0;1.0]
             *                  if random number is <= threshold than the cell will
             *                  be activated
             * @param mapper functor to map a block to a supercell
             */
            template<typename T_BoxWriteOnly, typename T_Mapping, typename T_Acc>
            DINLINE void operator()(
                T_Acc const& acc,
                T_BoxWriteOnly& buffWrite,
                uint32_t const seed,
                float const threshold,
                T_Mapping const& mapper) const
            {
                using namespace mappings::threads;

                using SuperCellSize = typename T_Mapping::SuperCellSize;
                constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;
                constexpr uint32_t numWorkers = T_numWorkers;
                uint32_t const workerIdx = cupla::threadIdx(acc).x;

                // get position in grid in units of SuperCells from blockID
                Space const block(mapper.getSuperCellIndex(Space(cupla::blockIdx(acc))));
                // convert position in unit of cells
                Space const blockCell = block * T_Mapping::SuperCellSize::toRT();
                // convert CUDA dim3 to DataSpace<DIM3>
                Space const threadIndex(cupla::threadIdx(acc));

                uint32_t const globalUniqueId = DataSpaceOperations<DIM2>::map(
                    mapper.getGridSuperCells() * T_Mapping::SuperCellSize::toRT(),
                    blockCell + DataSpaceOperations<DIM2>::template map<SuperCellSize>(workerIdx));

                // create a random number state and generator
                using RngMethod = random::methods::XorMin<T_Acc>;
                using State = typename RngMethod::StateType;
                State state;
                RngMethod method;
                method.init(acc, state, seed, globalUniqueId);
                using Distribution = random::distributions::Uniform<float, RngMethod>;
                using Random = random::Random<Distribution, RngMethod, State*>;
                Random rng(&state);

                ForEachIdx<IdxConfig<cellsPerSuperCell, numWorkers>>{
                    workerIdx}([&](uint32_t const linearIdx, uint32_t const) {
                    // cell index within the superCell
                    DataSpace<DIM2> const cellIdx = DataSpaceOperations<DIM2>::template map<SuperCellSize>(linearIdx);
                    // write 1(white) if uniform random number 0<rng<1 is smaller than 'threshold'
                    buffWrite(blockCell + cellIdx) = static_cast<bool>(rng(acc) <= threshold);
                });
            }
        };
    } // namespace kernel

    template<typename T_MappingDesc>
    struct Evolution
    {
        std::unique_ptr<T_MappingDesc> mapping;
        uint32_t rule;

        Evolution(uint32_t rule) : rule(rule)
        {
        }

        void init(Space const& layout, Space const& guardSize)
        {
            mapping = std::make_unique<T_MappingDesc>(layout, guardSize);
        }

        template<typename DBox>
        void initEvolution(DBox const& writeBox, float const fraction)
        {
            AreaMapping<CORE + BORDER, T_MappingDesc> mapper(*mapping);
            constexpr uint32_t numWorkers
                = traits::GetNumWorkers<math::CT::volume<typename T_MappingDesc::SuperCellSize>::type::value>::value;

            GridController<DIM2>& gc = Environment<DIM2>::get().GridController();
            uint32_t seed = gc.getGlobalSize() + gc.getGlobalRank();

            PMACC_KERNEL(kernel::RandomInit<numWorkers>{})
            (mapper.getGridDim(), numWorkers)(writeBox, seed, fraction, mapper);
        }

        template<uint32_t Area, typename DBox>
        void run(DBox const& readBox, DBox const& writeBox)
        {
            AreaMapping<Area, T_MappingDesc> mapper(*mapping);
            constexpr uint32_t numWorkers
                = traits::GetNumWorkers<math::CT::volume<typename T_MappingDesc::SuperCellSize>::type::value>::value;

            PMACC_KERNEL(kernel::Evolution<numWorkers>{})
            (mapper.getGridDim(), numWorkers)(readBox, writeBox, rule, mapper);
        }
    };

} // namespace gol
