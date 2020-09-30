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

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include "picongpu/particles/atomicPhysics/GetRealKineticEnergy.hpp"

#include <cstdint>


namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{

    template<
        typename T_Acc,
        typename T_Electron,
        typename T_Histogram
    >
    DINLINE void processElectron(
        T_Acc const & acc,
        T_Electron electron,
        T_Histogram const & histogram
    )
    {
        // todo: compute, probably via a generic algorithm
        float_X const energy = picongpu::particles::atomicPhysics::GetRealKineticEnergy( electron ) /
            picongpu::SI::ATOMIC_ENERGY_UNIT; // unit: ATOMIC_ENERGY_UNIT
        float_X const weight = electron[ weight_ ]; // unitless

        // look up in the histogram, which bin is this energy
        uint16_t const binIndex = histogram.getBinIndex( energy );

        // this could happen only if histogram did not have enough memory
        if( binIndex == histogram.getMaxNumberBins( ) )
            return;

        float_X const weightBin = histogram.getWeightBin( index );  // unitless
        float_X const deltaEnergyBin = histogram.getDeltaEnergyBin( index ); // unit: ATOMIC_ENERGY_UNIT

        float_X const deltaEnergy = deltaEnergyBin * weight / weightBin; // unit:: ATOMIC_ENERGY_UNIT

        float_X const scalingFactor = 1._X - deltaEnergy / energy ; // unitless
        electron[ momentum_ ] *= scalingFactor;
    }

    // Fill the histogram return via the last parameter
    // should be called inside the AtomicPhysicsKernel
    template<
        uint32_t T_numWorkers,
        typename T_Acc,
        typename T_Mapping,
        typename T_ElectronBox,
        typename T_Histogram
    >
    DINLINE void decelerateElectrons(
        T_Acc const & acc,
        T_Mapping mapper,
        T_ElectronBox electronBox,
        T_Histogram const & histogram
    )
    {
        using namespace mappings::threads;

        //// todo: express framesize better, not via supercell size
        constexpr uint32_t frameSize = pmacc::math::CT::volume< SuperCellSize >::type::value;
        constexpr uint32_t numWorkers = T_numWorkers;
        using ParticleDomCfg = IdxConfig<
            frameSize,
            numWorkers
        >;

        uint32_t const workerIdx = cupla::threadIdx( acc ).x;

        pmacc::DataSpace< simDim > const supercellIdx(
            mapper.getSuperCellIndex( DataSpace< simDim >( cupla::blockIdx( acc ) ) )
        );

        ForEachIdx<
            IdxConfig<
                1,
                numWorkers
            >
        > onlyMaster{ workerIdx };

        auto frame = electronBox.getLastFrame( supercellIdx );
        auto particlesInSuperCell = electronBox.getSuperCell( supercellIdx ).getSizeLastFrame( );

        // go over frames
        while( frame.isValid( ) )
        {
            // parallel loop over all particles in the frame
            ForEachIdx< ParticleDomCfg >{ workerIdx }
            (
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    // todo: check whether this if is necessary
                    if( linearIdx < particlesInSuperCell )
                    {
                        auto particle = frame[ linearIdx ];
                        processElectron(
                            acc,
                            particle,
                            histogram
                        );
                    }
                }
            );

            frame = electronBox.getPreviousFrame( frame );
            particlesInSuperCell = frameSize;
        }

    }

} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
