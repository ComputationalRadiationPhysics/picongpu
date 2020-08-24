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
        typename T_Ion,
        typename T_Histogram
    >
    DINLINE void processIon(
        T_Acc const & acc,
        RandomGenInt randomGenInt,
        RandomGenFloat randomGenFloat,
        T_Ion ion,
        T_Histogram * histogram
    )
    {
        // workaround: the types may be obtained in a better fashion
        auto configNumber = ion[ atomicConfigNumber_ ];
        using ConfigNumber = decltype( configNumber );
        using ConfigNumberDataType = decltype( ion[ atomicConfigNumber_ ].configNumber );

        float_X timeRemainingSI = picongpu::SI::DELTA_T_SI;
        while ( timeRemainingSI > 0.0_X )
        {
            ConfigNumberDataType newState = randomGenInt() %
                ConfigNumber::numberStates();
            // take a random bin existing in the histogram
            uint32_t histogramIndex = randomGenInt() %
                histogram->numBins;

            // TODO: implement rate matrix calculation
            float_X rateSI = 1.0_X;
            float_X deltaEnergy = 0.0_X;
            // TODO: compute rate matrix
            // to get rateSI and deltaE
            float_X probability = rateSI * timeRemainingSI;
            if ( probability >= 1.0_X )
            {
                ion[ atomicConfigNumber_ ].configNumber = newState;
                timeRemainingSI -= 1.0_X / rateSI;
            }
            else
                if ( randomGenFloat() <= probability )
                {
                    // note: perhaps there is a mix between
                    // ConfigNumber.configNumber and just ConfigNumber
                    ion[ atomicConfigNumber_ ].configNumber = newState;
                    timeRemainingSI = 0.0_X;
                }
        }
    }

    // Fill the histogram return via the last parameter
    // should be called inside the AtomicPhysicsKernel
    template<
        uint32_t T_numWorkers,
        typename T_Acc,
        typename T_Mapping,
        typename T_IonBox,
        typename T_Histogram
    >
    DINLINE void solveRateEquation(
        T_Acc const & acc,
        T_Mapping mapper,
        RngFactoryInt rngFactoryInt,
        RngFactoryFloat rngFactoryFloat,
        T_IonBox ionBox,
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
                        processIon(
                            acc,
                            generatorInt,
                            generatorFloat,
                            particle,
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
