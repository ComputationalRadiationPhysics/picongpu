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

    /** actual rate equation solver
     *
     * basic steps:
     * so long as time is remaining
     *  1.) choose a new state by choosing a random integer in the atomic state index
     *      range.
     *  2.) choose a random bin of energy histogram of electrons to interact with
     *  3.) calculate rate of change into this new state, with choosen electron energy
     *  3.) calculate the quasiProbability = rate * dt
     *  4.) if (quasiProbability > 1):
     *      - change ion atomic state to new state
     *      - reduce time by 1/rate, mean time between such changes
     *      - start again at 1.)
     *     else:
     *      if ( quasiProbability < 0 ):
     *          - start again at 1.)
     *      else:
     *          - decide randomly with quasiProbability if change to new state
     *          if we change state:
     *              - change ion state
     *  5.) finish
     */
    template<
        typename T_AtomicRate,
        typename T_Acc,
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
        // TODO: relace with better version
        auto configNumber = ion[ atomicConfigNumber_ ];
        using ConfigNumber = decltype( configNumber );
        using ConfigNumberDataType = decltype( ion[ atomicConfigNumber_ ].getStateIndex( ) ); // ? shorten

        using AtomicRate = T_AtomicRate;

        ConfigNumberDataType oldState;
        ConfigNumberDataType newState;
        uint32_t newStatesCollectionIndex;

        uint16_t histogramIndex;
        float_X energyElectron;
        float_X energyElectronBinWidth;

        float_X densityElectrons;

        float_X rate_SI;
        float_X deltaEnergy;
        float_X quasiProbability;


        // set remaining time to pic time step at the beginning
        float_X timeRemaining_SI = picongpu::SI::DELTA_T_SI;

        while ( timeRemaining_SI > 0.0_X )
        {
            // read out old state index
            oldState = configNumber.getStateIndex( );


            // get a random new state index, ?checkdatentyp randomIntGen
            newStatesCollectionIndex = randomGenInt( ) % atomicDataBox.getNumStates( );
            newState = atomicDataBox.getAtomicStateConfigNumberIndex( newStatesCollectionIndex );

            // newState = randomGenInt( ) % ConfigNumber::numberStates( );
            // unperformant since many states do not actually exist
            // and very large number of states possible in uint64 >> 15000

            // choose random histogram collection index
            histogramIndex = static_cast< uint16_t >( randomGenInt( ) ) %
                histogram->getNumBins( );

            // get energy of histogram bin with this collection index
            energyElectron = histogram->getEnergyBin(
                acc,
                histogramIndex,
                atomicDataBox
                ); // unit: ATOMIC_UNIT_ENERGY

            // get width of histogram bin with this collection index
            energyElectronBinWidth = histogram->getBinWidth(
                acc,
                true,   // answer to question: directionPositive?
                histogram->getLeftBoundaryBin( histogramIndex ), // unit: ATOMIC_UNIT_ENERGY
                histogram->getInitialGridWidth( ), // unit: ATOMIC_UNIT_ENERGY
                atomicDataBox
                );

            constexpr float_64 UNIT_VOLUME = UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH;
            // calculate density of electrons based on weight of electrons in this bin
            densityElectrons =  histogram->getWeightBin( histogramIndex ) /
                ( picongpu::numCellsPerSuperCell * picongpu::CELL_VOLUME * UNIT_VOLUME *
                    energyElectronBinWidth * picongpu::SI::ATOMIC_UNIT_ENERGY
                    );
            // (weighting * #/weighting) /
            //      ( numCellsPerSuperCell * Volume * m^3/Volume * AU * J/AU )
            // = # / (m^3 * J) => unit: 1/(m^3 * J), SI

            if ( oldState == newState )
            {
            // calculating quasiProbability for special case of keeping in current state

                // R_(i->i) = - sum_f( R_(i->f), rate_SI = - R_(i->i),
                // R ... rate, i ... initial state, f ... final state
                rate_SI = AtomicRate::totalRate(
                    acc,
                    oldState,   // unitless
                    energyElectron,     // unit: ATOMIC_UNIT_ENERGY
                    energyElectronBinWidth, // unit: ATOMIC_UNIT_ENERGY
                    densityElectrons,   // unit: 1/(m^3*J), SI
                    atomicDataBox
                    ); // unit: 1/s, SI

                quasiProbability = 1._X - rate_SI * timeRemaining_SI;
                deltaEnergy = 0._X;
            }
            else
            {
                // calculating quasiProbability for standard case of different newState

                rate_SI = AtomicRate::Rate(
                    acc,
                    oldState,   // unitless
                    newState,   // unitless
                    energyElectron,     // unit: ATOMIC_UNIT_ENERGY
                    energyElectronBinWidth, // unit: ATOMIC_UNIT_ENERGY
                    densityElectrons,   // unit: 1/(m^3*J), SI
                    atomicDataBox
                    ); // unit: 1/s, SI

                // get the change of electron energy
                deltaEnergy = AtomicRate::energyDifference(
                    acc,
                    oldState,
                    newState,
                    atomicDataBox
                    ) * ion[ weighting_ ] *
                    picongpu::particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE /
                    picongpu::SI::ATOMIC_UNIT_ENERGY;
                // J / (J/AU) => unit: ATOMIC_UNIT_ENERGY

                quasiProbability = rate_SI * timeRemaining_SI;
            }


            if ( quasiProbability >= 1.0_X )
            {
                // case: more than one change per time remaining
                // -> change once and reduce time remaining by mean time between such transitions
                //  can only happen in the case of newState != olstate, since otherwise 1 - ( >0 ) < 1

                // change atomic state of ion
                ion[ atomicConfigNumber_ ] = newState;

                // reduce time remaining
                if ( rate_SI > 0 )
                {
                    timeRemaining_SI -= 1.0_X / rate_SI;
                }

                // record energy removed or added to electrons
                histogram->removeEnergyFromBin(
                    acc,
                    histogramIndex, // unitless
                    deltaEnergy // unit: ATOMIC_UNIT_ENERGY 
                    );
            }
            else
            {
                if ( quasiProbability < 0._X )
                {
                    // case: newState != oldState
                    // quasiProbability can only be > 0, since AtomicRate::Rate( )>0
                    // and timeRemaining > 0

                    // case: newState == oldState
                    // on average change from original state into new more than once
                    // in timeRemaining
                    // => can not remain in current state -> must choose new state
                }
                else if ( randomGenFloat() <= quasiProbability )
                {
                    // case change only possible once
                    // => randomly change to newState in time remaining

                    // change ion state
                    ion[ atomicConfigNumber_ ] = newState;

                    // complete timeRemaining used
                    timeRemaining_SI = 0.0_X;

                    // record energy removed or added to electrons
                    histogram->removeEnergyFromBin(
                        acc,
                        histogramIndex, // unitless
                        deltaEnergy // unit: ATOMIC_UNIT_ENERGY
                        );
                }
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
