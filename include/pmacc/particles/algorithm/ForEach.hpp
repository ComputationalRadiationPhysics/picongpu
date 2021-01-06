/* Copyright 2017-2021 Axel Huebl, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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

#include "pmacc/Environment.hpp"
#include "pmacc/particles/frame_types.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"
#include "pmacc/mappings/threads/ForEachIdx.hpp"
#include "pmacc/mappings/threads/IdxConfig.hpp"
#include "pmacc/mappings/threads/WorkerCfg.hpp"

#include <utility>


namespace pmacc
{
    namespace particles
    {
        namespace algorithm
        {
            namespace acc
            {
                namespace detail
                {
                    /** operate on particles of a species
                     *
                     * @tparam T_numWorkers number of workers
                     */
                    template<uint32_t T_numWorkers>
                    struct ForEachParticle
                    {
                        /** operate on particles
                         *
                         * @tparam T_Acc alpaka accelerator type
                         * @tparam T_Functor type of the functor to operate on a particle
                         * @tparam T_Mapping mapping functor type
                         * @tparam T_ParBox pmacc::ParticlesBox, type of the species box
                         *
                         * @param acc alpaka accelerator
                         * @param functor functor to operate on a particle
                         *                must fulfill the interface pmacc::functor::Interface<F, 1u, void>
                         * @param mapper functor to map a block to a supercell
                         * @param pb particles species box
                         */
                        template<typename T_Acc, typename T_Functor, typename T_Mapping, typename T_ParBox>
                        DINLINE void operator()(
                            T_Acc const& acc,
                            T_Functor functor,
                            T_Mapping const mapper,
                            T_ParBox pb) const
                        {
                            using namespace mappings::threads;

                            using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                            constexpr uint32_t dim = SuperCellSize::dim;
                            constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                            constexpr uint32_t numWorkers = T_numWorkers;

                            uint32_t const workerIdx = cupla::threadIdx(acc).x;

                            DataSpace<dim> const superCellIdx(
                                mapper.getSuperCellIndex(DataSpace<dim>(cupla::blockIdx(acc))));

                            auto const& superCell = pb.getSuperCell(superCellIdx);
                            uint32_t const numPartcilesInSupercell = superCell.getNumParticles();


                            // end kernel if we have no particles
                            if(numPartcilesInSupercell == 0)
                                return;

                            using FramePtr = typename T_ParBox::FramePtr;
                            FramePtr frame = pb.getFirstFrame(superCellIdx);

                            // offset of the superCell (in cells, without any guards) to the origin of the local domain
                            DataSpace<dim> const localSuperCellOffset = superCellIdx - mapper.getGuardingSuperCells();

                            auto accFunctor = functor(acc, localSuperCellOffset, WorkerCfg<T_numWorkers>{workerIdx});

                            for(uint32_t parOffset = 0; parOffset < numPartcilesInSupercell; parOffset += frameSize)
                            {
                                using ParticleDomCfg = IdxConfig<frameSize, numWorkers>;

                                // loop over all particles in the frame
                                ForEachIdx<ParticleDomCfg>{workerIdx}([&](uint32_t const linearIdx, uint32_t const) {
                                    // particle index within the supercell
                                    uint32_t parIdx = parOffset + linearIdx;
                                    auto particle = frame[linearIdx];

                                    bool const isPar = parIdx < numPartcilesInSupercell;
                                    if(isPar)
                                        accFunctor(acc, particle);
                                });

                                frame = pb.getNextFrame(frame);
                            }
                        }
                    };

                } // namespace detail
            } // namespace acc

            /** Run a unary functor for each particle of a species
             *
             * @warning Does NOT fill gaps automatically! If the
             *          operation deactivates particles or creates "gaps" in any
             *          other way, CallFillAllGaps needs to be called for the
             *          species manually afterwards!
             *
             * Operates on the domain CORE and BORDER
             *
             * @tparam T_Species type of the species
             * @tparam T_Functor unary particle functor type which follows the interface of
             *                   pmacc::functor::Interface<F, 1u, void>
             *
             * @param species species to operate on
             * @param functor operation which is applied to each particle of the species
             */
            template<typename T_Species, typename T_Functor>
            void forEach(T_Species&& species, T_Functor functor)
            {
                using MappingDesc = decltype(species.getCellDescription());
                AreaMapping<CORE + BORDER, MappingDesc> mapper(species.getCellDescription());

                using SuperCellSize = typename MappingDesc::SuperCellSize;

                constexpr uint32_t numWorkers
                    = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                PMACC_KERNEL(acc::detail::ForEachParticle<numWorkers>{})
                (mapper.getGridDim(), numWorkers)(std::move(functor), mapper, species.getDeviceParticlesBox());
            }

        } // namespace algorithm
    } // namespace particles
} // namespace pmacc
