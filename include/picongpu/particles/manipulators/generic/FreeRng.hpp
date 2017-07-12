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
#include <string>


namespace picongpu
{
namespace particles
{
namespace manipulators
{
namespace generic
{
namespace acc
{
    template<
        typename T_Functor,
        typename T_RngType
    >
    struct FreeRng : private T_Functor
    {

        using Functor = T_Functor;
        using RngType = T_RngType;

        DINLINE FreeRng(
            Functor const & functor,
            RngType const & rng
        ) :
            T_Functor( functor ), m_rng( rng )
        {
        }

        /** call user functor
         *
         * The random number generator is initialized with the first call.
         *
         * @param particle particle which is given to the user functor
         * @return void is used to enable the operator if the user functor except two arguments
         */
        template<
            typename T_Particle,
            typename ... T_Args
        >
        DINLINE
        void operator()(
            T_Particle& particle,
            T_Args && ... args
        )
        {
            namespace nvrng = nvidia::rng;

            Functor::operator()(
                m_rng,
                particle,
                args ...
            );
        }

    private:

        RngType m_rng;
    };
} // namespace acc

    template<
        typename T_Functor,
        typename T_Distribution,
        typename T_Seed,
        typename T_SpeciesType
    >
    struct FreeRng : private T_Functor
    {

        using Functor = T_Functor;
        using Distribution = T_Distribution;
        using SpeciesType = T_SpeciesType;

        using RngType = PMacc::nvidia::rng::RNG<
            nvidia::rng::methods::Xor,
            Distribution
        >;

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
        HINLINE FreeRng(
            uint32_t currentStep,
            typename std::enable_if<
                std::is_constructible<
                    DeferFunctor,
                    uint32_t
                >::value
            >::type* = 0
        ) : Functor( currentStep )
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
        HINLINE FreeRng(
            uint32_t currentStep,
            typename std::enable_if<
                std::is_constructible< DeferFunctor >::value
            >::type* = 0
        ) : Functor( )
        {
            hostInit( currentStep );
        }

        /** create functor for the accelerator
         *
         * @tparam T_WorkerCfg PMacc::mappings::threads::WorkerCfg, configuration of the worker
         * @param localSupercellOffset offset (in superCells, without any guards) relative
         *                        to the origin of the local domain
         * @param workerCfg configuration of the worker
         */
        template< typename T_WorkerCfg >
        DINLINE
        acc::FreeRng<
            Functor,
            RngType
        > operator()(
            DataSpace< simDim > const & localSupercellOffset,
            T_WorkerCfg const & workerCfg
        )
        {
            namespace nvrng = nvidia::rng;

            using FrameType = typename SpeciesType::FrameType;
            using SuperCellSize = typename FrameType::SuperCellSize;



            uint32_t const cellIdx = DataSpaceOperations< simDim >::map(
                localCells,
                localSupercellOffset * SuperCellSize::toRT( ) +
                    DataSpaceOperations< simDim >::template map< SuperCellSize >( workerCfg.getWorkerIdx( ) )
            );
            RngType const rng = nvrng::create(
                nvidia::rng::methods::Xor(
                    seed,
                    cellIdx
                ),
                Distribution{}
            );
            return acc::FreeRng<
                Functor,
                RngType
            >(
                *reinterpret_cast< Functor * >( this ),
                rng
            );
        }

        HINLINE std::string
        getName( ) const
        {
            return std::string("FreeRNG");
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
                T_Seed::value;
            /* mixing the final seed with the current time step to avoid
             * correlations between time steps
             */
            seed = seedPerRank( seed ) ^ currentStep;

            SubGrid< simDim > const & subGrid = Environment< simDim >::get().SubGrid();
            localCells = subGrid.getLocalDomain().size;
        }

        DataSpace< simDim > localCells;
        uint32_t seed;
    };

} // namepsace generic
} // namespace manipulators
} // namespace particles
} // namespace picongpu
