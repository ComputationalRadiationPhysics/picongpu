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

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include <cstdint>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            template<typename T_Worker, typename T_Electron, typename T_Histogram, typename T_AtomicDataBox>
            DINLINE void processElectron(
                T_Worker const& worker,
                T_Electron electron,
                T_Histogram const& histogram,
                T_AtomicDataBox atomicDataBox)
            {
                /// @todo : choose algorithm by particle? @BrianMarre, 2021
                float_64 const energyPhysicalElectron
                    = picongpu::particles::atomicPhysics::GetRealKineticEnergy::KineticEnergy(electron)
                    / picongpu::SI::ATOMIC_UNIT_ENERGY; // unit: ATOMIC_UNIT_ENERGY

                // look up in the histogram, which bin corresponds to this energy
                uint16_t binIndex = histogram.getBinIndex(
                    worker,
                    energyPhysicalElectron, // unit: ATOMIC_UNIT_ENERGY
                    atomicDataBox);

                // case: electron missing from histogram due to not enough histogram
                // bins/too few intermediate bins
                if(binIndex == histogram.getMaxNumberBins())
                    return;

                float_X const weightBin = histogram.getWeightBin(binIndex); // unitless
                float_X const deltaEnergyBin = histogram.getDeltaEnergyBin(binIndex);
                // unit: ATOMIC_UNIT_ENERGY

                /// @todo : create attribute functor for physical particle properties?, @BrianMarre, 2021
                constexpr float_64 c_SI = picongpu::SI::SPEED_OF_LIGHT_SI; // unit: m/s, SI
                float_64 m_e_rel = attribute::getMass(1.0_X, electron) * picongpu::UNIT_MASS * c_SI * c_SI
                    / picongpu::SI::ATOMIC_UNIT_ENERGY; // unit: ATOMIC_UNIT_ENERGY

                // distribute energy change as mean by weight on all electrons in bin
                float_64 newEnergyPhysicalElectron
                    = energyPhysicalElectron
                    + static_cast<float_64>(
                          deltaEnergyBin
                          / (static_cast<float_X>(picongpu::particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE)
                             * weightBin));
                // unit:: ATOMIC_UNIT_ENERGY

                // case: too much energy removed
                if(newEnergyPhysicalElectron < 0)
                    newEnergyPhysicalElectron = 0._X; // extract as much as possible, rest should be neglible

                float_64 newPhysicalElectronMomentum
                    = math::sqrt(newEnergyPhysicalElectron * (newEnergyPhysicalElectron + 2 * m_e_rel))
                    * picongpu::SI::ATOMIC_UNIT_ENERGY / c_SI;
                // AU = ATOMIC_UNIT_ENERGY
                // sqrt(AU * (AU + AU)) / (AU/J) / c = sqrt(AU^2)/(AU/J) / c = J/c = kg*m^2/s^2/(m/s)
                // unit: kg*m/s, SI

                float_X previousMomentumVectorLength2 = pmacc::math::l2norm2(electron[momentum_]);
                // unit: internal, scaled

                // case: not moving electron
                if(previousMomentumVectorLength2 == 0._X)
                    previousMomentumVectorLength2 = 1._X; // no need to resize 0-vector

                // if previous momentum == 0, discards electron,
                // @todo select random direction to apply momentum, Brian Marre, 2022
                electron[momentum_] *= 1 / previousMomentumVectorLength2 // get unity vector of momentum
                    * static_cast<float_X>(newPhysicalElectronMomentum
                                           * electron[weighting_] // new momentum scaled and in internal units
                                           / (picongpu::UNIT_MASS * picongpu::UNIT_LENGTH / picongpu::UNIT_TIME));
                // unit: internal units
            }

            // Fill the histogram return via the last parameter
            // should be called inside the AtomicPhysicsKernel
            template<
                typename T_Worker,
                typename T_Mapping,
                typename T_ElectronBox,
                typename T_Histogram,
                typename T_AtomicDataBox>
            DINLINE void decelerateElectrons(
                T_Worker const& worker,
                T_Mapping mapper,
                T_ElectronBox electronBox,
                T_Histogram const& histogram,
                T_AtomicDataBox atomicDataBox)
            {
                pmacc::DataSpace<simDim> const superCellIdx(
                    mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc()))));

                auto frame = electronBox.getLastFrame(superCellIdx);
                auto particlesInSuperCell = electronBox.getSuperCell(superCellIdx).getSizeLastFrame();

                constexpr uint32_t frameSize = T_ElectronBox::frameSize;

                auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize>(worker);

                // go over frames
                while(frame.isValid())
                {
                    // parallel loop over all particles in the frame
                    forEachParticleSlotInFrame(
                        [&](uint32_t const linearIdx)
                        {
                            if(linearIdx < particlesInSuperCell)
                            {
                                auto particle = frame[linearIdx];
                                processElectron(worker, particle, histogram, atomicDataBox);
                            }
                        });

                    worker.sync();

                    frame = electronBox.getPreviousFrame(frame);
                    particlesInSuperCell = frameSize;
                }
            }

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
