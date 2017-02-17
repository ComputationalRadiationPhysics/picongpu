/* Copyright 2015-2017 Rene Widera, Alexander Grund
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

#include "simulation_defines.hpp"
#include "nvidia/rng/RNG.hpp"
#include "nvidia/rng/methods/Xor.hpp"
#include "mpi/SeedPerRank.hpp"
#include "traits/GetUniqueTypeId.hpp"

#include <utility>
#include <type_traits>


namespace picongpu
{
namespace particles
{
namespace manipulators
{

    template<
        typename T_Functor,
        typename T_Distribution,
        typename T_SpeciesType
    >
    struct FreeRngImpl : private T_Functor
    {

        using Functor = T_Functor;
        using Distribution = T_Distribution;
        using SpeciesType = T_SpeciesType;
        using SpeciesName = typename MakeIdentifier<SpeciesType>::type;

        /** constructor
         *
         * This constructor is only compiled if the user functor has
         * a host side constructor with one (uint32_t) argument.
         *
         * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
         *                      the constructor
         * @param currentStep current simulation time step
         * @param is used to enable/disable the constructor (do not pass any value to this parameter)
         */
        template< typename DeferFunctor = Functor >
        HINLINE FreeRngImpl(
            uint32_t currentStep,
            typename std::enable_if<
                std::is_constructible<
                    DeferFunctor,
                    uint32_t
                >::value
            >::type* = 0
        ) : Functor( currentStep ), isInitialized( false )
        {
            hostInit( currentStep );
        }

        /** constructor
         *
         * This constructor is only compiled if the user functor has a default constructor.
         *
         * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
         *                      the constructor
         * @param currentStep simulation time step
         * @param is used to enable/disable the constructor (do not pass any value to this parameter)
         */
        template< typename DeferFunctor = Functor >
        HINLINE FreeRngImpl(
            uint32_t currentStep,
            typename std::enable_if<
                std::is_constructible< DeferFunctor >::value
            >::type* = 0
        ) : Functor( ), isInitialized( false )
        {
            hostInit( currentStep );
        }

        /** call user functor
         *
         * The random number generator is initialized with the first call.
         *
         * @param cell superCell index within the local volume
         * @param unused
         * @param isParticle1 define if the reference @p particleSpecies1 is valid
         * @param unused
         * @return void is used to enable the operator if the user functor except two arguments
         */
        template<
            typename T_Particle1,
            typename T_Particle2
        >
        DINLINE
        void operator()(
            DataSpace< simDim > const & localSuperCellOffset,
            T_Particle1& particle1,
            T_Particle2&,
            bool const isParticle1,
            bool const
        )
        {
            namespace nvrng = nvidia::rng;

            using FrameType = typename T_Particle1::FrameType;

            if( !isInitialized )
            {
                /** @todo: it is a wrong assumption that the threadIdx can be used to
                 * define the cell within the superCell. This is only allowed if we not
                 * use alpaka. We need to distinguish between manipulators those are working on the
                 * cell domain and on the particle domain.
                 */
                DataSpace< simDim > const threadIndex( threadIdx );
                uint32_t const cellIdx = DataSpaceOperations< simDim >::map(
                    localCells,
                    localSuperCellOffset + threadIndex
                );
                rng = nvrng::create(
                    nvidia::rng::methods::Xor(
                        seed,
                        cellIdx
                    ),
                    Distribution{}
                );
                isInitialized = true;
            }

            if( isParticle1 )
            {
                Functor::operator()(
                    rng,
                    particle1
                );
            }
        }

    private:

        /** initialize member variables
         *
         * set RNG seed and calculate simulation local size
         *
         * @param currentStep time step of the simulation
         */
        HINLINE
        void
        hostInit( uint32_t currentStep )
        {
            using FrameType = typename SpeciesType::FrameType;

            GlobalSeed globalSeed;
            mpi::SeedPerRank<simDim> seedPerRank;
            /* generate a global unique id by xor the
             *   - gpu id `globalSeed()`
             *   - a species id
             *   - a fix id for this functor `FREERNG_SEED` (defined in `seed.param`)
             */
            seed = globalSeed() ^
                PMacc::traits::GetUniqueTypeId<
                    FrameType,
                    uint32_t
                >::uid() ^
                FREERNG_SEED;
            /* mixing the final seed with the current time step to avoid
             * correlations between time steps
             */
            seed = seedPerRank( seed ) ^ currentStep;

            SubGrid< simDim > const & subGrid = Environment< simDim >::get().SubGrid();
            localCells = subGrid.getLocalDomain().size;
        }

        using RngType = PMacc::nvidia::rng::RNG<
            nvidia::rng::methods::Xor,
            Distribution
        >;
        RngType rng;
        DataSpace< simDim > localCells;
        bool isInitialized;
        uint32_t seed;
    };

} //namespace manipulators
} //namespace particles
} //namespace picongpu
