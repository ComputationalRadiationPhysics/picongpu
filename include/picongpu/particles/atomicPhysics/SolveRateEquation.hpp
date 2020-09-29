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


#include "picongpu/particles/particleToGrid/derivedAttributes/Density.hpp"

#include <cstdint>


namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{

    // for now 32-bit hardcoded, that should be
    // underlying type of config number
    using DistributionInt = pmacc::random::distributions::Uniform< uint32_t >;
    using RngFactoryInt = particles::functor::misc::Rng< DistributionInt >;
    using DistributionFloat = pmacc::random::distributions::Uniform< float_X >;
    using RngFactoryFloat = particles::functor::misc::Rng< DistributionFloat >;
    using RandomGenInt = RngFactoryInt::RandomGen;
    using RandomGenFloat = RngFactoryFloat::RandomGen;

    template<
        typename T_Acc,
        typename T_AtomicRate,
        typename T_Ion,
        typename T_AtomicDataBox,
        typename T_Histogram
    >
    DINLINE void processIon(
        T_Acc const & acc,
        RandomGenInt randomGenInt,
        RandomGenFloat randomGenFloat,
        T_Ion ion,
        T_AtomicDataBox const atomicDataBox,
        T_Histogram * histogram
    )
    {
        // workaround: the types may be obtained in a better fashion
        auto configNumber = ion[ atomicConfigNumber_ ];
        using ConfigNumber = decltype( configNumber );
        using ConfigNumberDataType = decltype( ion[ atomicConfigNumber_ ].configNumber );

        float_X timeRemaining_SI = picongpu::SI::DELTA_T_SI;

        using AtomicRate = T_AtomicRate;

        while ( timeRemaining_SI > 0.0_X )
        {
            // get a random new state
            ConfigNumberDataType newState = randomGenInt() %
                ConfigNumber::numberStates();
            // take a random bin existing in the histogram
            uint16_t histogramIndex = static_cast< uint16_t >( randomGenInt( ) ) %
                histogram->getNumBins();

            // TODO: implement rate matrix calculation
            ConfigNumberDataType oldState = ion[ atomicConfigNumber_ ];
            float_X energyElectron = histogram->getEnergyBin( histogramIndex ); // unit: ATOMIC_ENERGY_UNIT
            float_X energyElectronBinWidth = histogram->getBinWidth(
                true,
                histogram->getLeftBoundaryBin( histogramIndex ), // unit: ATOMIC_ENERGY_UNIT
                histogram->getInitialGridWidth( ) // unit: ATOMIC_ENERGY_UNIT
                );
            // (weighting * #/weighting) /
            //      ( numCellsPerSuperCell * Volume * m^3/Volume * AU * J/AU )
            // = # / (m^3 * J)
            float_X densityElectrons = ( histogram->getWeights( histogramIndex ) *
                picongpu::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE ) /
                ( picongpu::numCellsPerSuperCell * picongpu::CELL_VOLUME * UNIT_VOLUME *
                    energyElectronBinWidth * picongpu::SI::ATOMIC_ENERGY_UNIT );
                // unit: 1/(m^3 * J), SI

            float_X rate_SI = AtomicRate::Rate(
                oldState,   // unitless
                newState,   // unitless
                energyElectron,     // unit: ATOMIC_ENERGY_UNIT
                energyElectronBinWidth, // unit: ATOMIC_ENERGY_UNIT
                densityElectrons,   // unit: 1/(m^3*J), SI
                atomicDataBox
                ); // unit: 1/s, SI

            // J / J/AU
            float_X deltaEnergy = energyDifference(
                oldState,
                newState,
                atomicDataBox
                ) * ion[ weighting_ ] *
                picongpu::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE /
                picongpu::SI::ATOMIC_ENERGY_UNIT; // unit: ATOMIC_ENERGY_UNIT

            float_X quasiProbability = rate_SI * timeRemaining_SI;
            if ( quasiProbability >= 1.0_X )
            {
                ion[ atomicConfigNumber_ ].configNumber = newState;
                timeRemainingSI -= 1.0_X / rateSI;
                histogram->removeEnergyFromBin(
                    histogramIndex, // unitless
                    deltaEnergy // unit: ATOMIC_ENERGY_UNIT
                    );
            }
            else
                if ( randomGenFloat() <= quasiProbability )
                {
                    // note: perhaps there is a mix between
                    // ConfigNumber.configNumber and just ConfigNumber
                    ion[ atomicConfigNumber_ ].configNumber = newState;
                    timeRemaining_SI = 0.0_X;
                    histogram->removeEnergyFromBin(
                        histogramIndex, // unitless
                        deltaEnergy // unit: ATOMIC_ENERGY_UNIT
                    );
                }
        }
    }

    // Fill the histogram return via the last parameter
    // should be called inside the AtomicPhysicsKernel
    template<
        uint32_t T_numWorkers,
        typename T_AtomicRate,
        typename T_Acc,
        typename T_Mapping,
        typename T_IonBox,
        typename T_AtomicDataBox,
        typename T_Histogram
    >
    DINLINE void solveRateEquation(
        T_Acc const & acc,
        T_Mapping mapper,
        RngFactoryInt rngFactoryInt,
        RngFactoryFloat rngFactoryFloat,
        T_IonBox ionBox,
        T_AtomicDataBox const atomicDataBox,
        T_Histogram * histogram
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

        auto frame = ionBox.getLastFrame( supercellIdx );
        auto particlesInSuperCell = ionBox.getSuperCell( supercellIdx ).getSizeLastFrame( );

        pmacc::mappings::threads::WorkerCfg<numWorkers> workerCfg( workerIdx );
        auto generatorInt = rngFactoryInt( acc, supercellIdx, workerCfg );
        auto generatorFloat = rngFactoryFloat( acc, supercellIdx, workerCfg );

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
                        processIon< T_AtomicRate >(
                            acc,
                            generatorInt,
                            generatorFloat,
                            particle,
                            atomicDataBox,
                            histogram
                        );
                    }
                }
            );

            cupla::__syncthreads( acc );

            frame = ionBox.getPreviousFrame( frame );
            particlesInSuperCell = frameSize;
        }

    }

} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
