/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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

#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
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
                typename T_Worker,
                typename T_ElectronBox,
                typename T_Mapping,
                typename T_Histogram,
                typename T_AtomicDataBox>
            DINLINE void fillHistogram(
                T_Worker const& worker,
                T_ElectronBox const electronBox,
                T_Mapping mapper,
                T_Histogram* histogram,
                T_AtomicDataBox atomicDataBox)
            {
                pmacc::DataSpace<simDim> const superCellIdx(
                    mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc()))));

                auto frame = electronBox.getLastFrame(superCellIdx);
                auto particlesInSuperCell = electronBox.getSuperCell(superCellIdx).getSizeLastFrame();

                constexpr uint32_t frameSize = T_ElectronBox::frameSize;

                auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize>(worker);
                auto onlyMaster = lockstep::makeMaster(worker);

                // go over frames using common histogram
                while(frame.isValid())
                {
                    // parallel loop over all particles in the frame
                    forEachParticleSlotInFrame(
                        [&](uint32_t const linearIdx)
                        {
                            if(linearIdx < particlesInSuperCell)
                            {
                                auto particle = frame[linearIdx];

                                /// @todo : make this configurable, Brian Marre, 2021
                                float_64 const energy_SI = GetRealKineticEnergy::KineticEnergy(particle);
                                // unit: J, SI

                                histogram->binObject(
                                    worker,
                                    static_cast<float_X>(
                                        energy_SI / picongpu::SI::ATOMIC_UNIT_ENERGY), // unit: ATOMIC_UNIT_ENERGY
                                    particle[weighting_],
                                    atomicDataBox);
                            }
                        });

                    // A single thread does bookkeeping
                    worker.sync();
                    onlyMaster([&]() { histogram->updateWithNewBins(); });
                    worker.sync();

                    frame = electronBox.getPreviousFrame(frame);
                    particlesInSuperCell = frameSize;
                }
            }

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
