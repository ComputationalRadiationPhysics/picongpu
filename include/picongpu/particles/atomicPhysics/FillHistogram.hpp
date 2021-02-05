/* Copyright 2013-2020 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov,
 *                     Brian Marre
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
#include "picongpu/particles/atomicPhysics/GetRealKineticEnergy.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            // Fill the histogram return via the last parameter
            // should be called inside the AtomicPhysicsKernel
            template<
                uint32_t T_numWorkers,
                typename T_Acc,
                typename T_ElectronBox,
                typename T_Mapping,
                typename T_Histogram,
                typename T_AtomicDataBox>
            DINLINE void fillHistogram(
                T_Acc const& acc,
                T_ElectronBox const electronBox,
                T_Mapping mapper,
                T_Histogram* histogram,
                T_AtomicDataBox atomicDataBox)
            {
                //{ preparations
                using namespace mappings::threads;

                // TODO: express framesize better, not via supercell size
                constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                constexpr uint32_t numWorkers = T_numWorkers;
                using ParticleDomCfg = IdxConfig<frameSize, numWorkers>;

                uint32_t const workerIdx = cupla::threadIdx(acc).x;

                pmacc::DataSpace<simDim> const supercellIdx(
                    mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(acc))));

                ForEachIdx<IdxConfig<1, numWorkers>> onlyMaster{workerIdx};

                auto frame = electronBox.getLastFrame(supercellIdx);
                auto particlesInSuperCell = electronBox.getSuperCell(supercellIdx).getSizeLastFrame();
                //}


                // go over frames
                while(frame.isValid())
                {
                    // parallel loop over all particles in the frame
                    ForEachIdx<ParticleDomCfg>{workerIdx}([&](uint32_t const linearIdx, uint32_t const) {
                        // TODO: check whether this if is necessary
                        if(linearIdx < particlesInSuperCell)
                        {
                            // NOTE: all particle[ ... ] returns in PIC units, not SI
                            // note: there is UNIT_ENERGY that can help with conversion
                            // note3: maybe getEnergy could become a generic algorithm
                            auto const particle = frame[linearIdx];

                            float_X const energy_SI
                                = picongpu::particles::atomicPhysics::GetRealKineticEnergy::KineticEnergy(particle);
                            // unit: J, SI

                            histogram->binObject(
                                acc,
                                energy_SI / picongpu::SI::ATOMIC_UNIT_ENERGY, // unit: ATOMIC_UNIT_ENERGY
                                particle[weighting_],
                                atomicDataBox);
                        }
                    });

                    // A single thread does bookkeeping
                    cupla::__syncthreads(acc);
                    onlyMaster([&](uint32_t const, uint32_t const) { histogram->updateWithNewBins(); });
                    cupla::__syncthreads(acc);

                    frame = electronBox.getPreviousFrame(frame);
                    particlesInSuperCell = frameSize;
                }
            }

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
