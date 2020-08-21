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
        typename T_Histogram
    >
    DINLINE void fillHistogram(
        T_Acc const & acc,
        T_ElectronBox const electronBox,
        T_Mapping mapper,
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

        // relative offset (in cells) to the local domain start (including the guard)
        DataSpace< simDim > const superCellOffset = supercellIdx * SuperCellSize::toRT();



        ForEachIdx<
            IdxConfig<
                1,
                numWorkers
            >
        > onlyMaster{ workerIdx };

        auto frame = electronBox.getLastFrame( superCellOffset );
        auto particlesInSuperCell = electronBox.getSuperCell( superCellOffset ).getSizeLastFrame( );

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
                        // NOTE: all particle[ ... ] returns in PIC units, not SI
                        // note: there is UNIT_ENERGY that can help with conversion
                        // note 3: maybe getEnergy could become a generic algorithm
                        auto const particle = frame[ linearIdx ];
                        float_X const m = attribute::getMass(1.0_X, particle); //particle[ massRatio_ ] * SI::BASE_MASS;     //Unit: kg
                        constexpr auto c = SI::SPEED_OF_LIGHT_SI;                   //Unit: m/s

                        float3_X vectorP = particle[ momentum_ ];
                        // we probably have a math function for ||p||^2
                        float_X pSquared = math::abs2( vectorP ); /* vectorP[0]*vectorP[0] +
                            vectorP[1]*vectorP[1] +
                            vectorP[2]*vectorP[2]; */                     //unit:kg*m/s

                        // note about math functions:
                        // in the dev branch need to add pmacc:: and acc as first parameter

                        //unit: kg*m^2/s^2 = Nm
                        auto const energy = math::sqrt(
                                m*m * c*c*c*c + pSquared * c*c
                        );
                        histogram->binObject(
                            acc,
                            energy,
                            particle[ weighting_ ]
                        );
                    }
                }
            );

            // A single thread does bookkeeping
            cupla::__syncthreads( acc );
            onlyMaster(
                [&](
                    uint32_t const,
                    uint32_t const
                )
                {
                    histogram->updateWithNewBins();
                }
            );
            cupla::__syncthreads( acc );

            frame = electronBox.getPreviousFrame( frame );
            particlesInSuperCell = frameSize;
        }

    }

} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
