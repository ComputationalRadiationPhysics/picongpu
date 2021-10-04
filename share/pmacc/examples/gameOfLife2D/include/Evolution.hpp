/* Copyright 2013-2023 Rene Widera, Marco Garten
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

#include <pmacc/dimensions/DataSpaceOperations.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/threads/ThreadCollective.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/boxes/CachedBox.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>
#include <pmacc/random/Random.hpp>
#include <pmacc/random/distributions/distributions.hpp>
#include <pmacc/random/methods/methods.hpp>

#include <memory>

namespace gol
{
    namespace kernel
    {
        using namespace pmacc;

        /** run game of life stencil
         *
         * evaluate each cell in the supercell
         */
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
            template<typename T_BoxReadOnly, typename T_BoxWriteOnly, typename T_Mapping, typename T_Worker>
            DINLINE void operator()(
                T_Worker const& worker,
                T_BoxReadOnly const& buffRead,
                T_BoxWriteOnly buffWrite,
                uint32_t const rule,
                T_Mapping const& mapper) const
            {
                using Type = typename T_BoxReadOnly::ValueType;
                using SuperCellSize = typename T_Mapping::SuperCellSize;
                using BlockArea = SuperCellDescription<SuperCellSize, math::CT::Int<1, 1>, math::CT::Int<1, 1>>;
                auto cache = CachedBox::create<0, SharedDataBoxMemoryLayout, Type>(worker, BlockArea());

                Space const block(mapper.getSuperCellIndex(Space(cupla::blockIdx(worker.getAcc()))));
                Space const blockCell = block * T_Mapping::SuperCellSize::toRT();

                constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

                auto buffRead_shifted = buffRead.shift(blockCell);

                auto collective = makeThreadCollective<BlockArea>();

                math::operation::Assign assign;
                collective(worker, assign, cache, buffRead_shifted);

                worker.sync();

                lockstep::makeForEach<cellsPerSuperCell>(worker)(
                    [&](uint32_t const linearIdx)
                    {
                        // cell index within the superCell
                        DataSpace<DIM2> const cellIdx
                            = DataSpaceOperations<DIM2>::template map<SuperCellSize>(linearIdx);

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
         */

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
            template<typename T_BoxWriteOnly, typename T_Mapping, typename T_Worker>
            DINLINE void operator()(
                T_Worker const& worker,
                T_BoxWriteOnly buffWrite,
                uint32_t const seed,
                float const threshold,
                T_Mapping const& mapper) const
            {
                using SuperCellSize = typename T_Mapping::SuperCellSize;
                constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

                // get position in grid in units of SuperCells from blockID
                Space const block(mapper.getSuperCellIndex(Space(cupla::blockIdx(worker.getAcc()))));
                // convert position in unit of cells
                Space const blockCell = block * T_Mapping::SuperCellSize::toRT();

                uint32_t const globalUniqueId = DataSpaceOperations<DIM2>::map(
                    mapper.getGridSuperCells() * T_Mapping::SuperCellSize::toRT(),
                    blockCell + DataSpaceOperations<DIM2>::template map<SuperCellSize>(worker.getWorkerIdx()));

                // create a random number state and generator
                using RngMethod = random::methods::XorMin<typename T_Worker::Acc>;
                using State = typename RngMethod::StateType;
                State state;
                RngMethod method;
                method.init(worker, state, seed, globalUniqueId);
                using Distribution = random::distributions::Uniform<float, RngMethod>;
                using Random = random::Random<Distribution, RngMethod, State*>;
                Random rng(&state);

                lockstep::makeForEach<cellsPerSuperCell>(worker)(
                    [&](uint32_t const linearIdx)
                    {
                        // cell index within the superCell
                        DataSpace<DIM2> const cellIdx
                            = DataSpaceOperations<DIM2>::template map<SuperCellSize>(linearIdx);
                        // write 1(white) if uniform random number 0<rng<1 is smaller than 'threshold'
                        buffWrite(blockCell + cellIdx) = static_cast<bool>(rng(worker) <= threshold);
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

            GridController<DIM2>& gc = Environment<DIM2>::get().GridController();
            uint32_t seed = gc.getGlobalSize() + gc.getGlobalRank();

            auto workerCfg = lockstep::makeWorkerCfg(typename T_MappingDesc::SuperCellSize{});
            PMACC_LOCKSTEP_KERNEL(kernel::RandomInit{}, workerCfg)
            (mapper.getGridDim())(writeBox, seed, fraction, mapper);
        }

        template<uint32_t Area, typename DBox>
        void run(DBox const& readBox, DBox const& writeBox)
        {
            AreaMapping<Area, T_MappingDesc> mapper(*mapping);
            auto workerCfg = lockstep::makeWorkerCfg(typename T_MappingDesc::SuperCellSize{});
            PMACC_LOCKSTEP_KERNEL(kernel::Evolution{}, workerCfg)
            (mapper.getGridDim())(readBox, writeBox, rule, mapper);
        }
    };

} // namespace gol
