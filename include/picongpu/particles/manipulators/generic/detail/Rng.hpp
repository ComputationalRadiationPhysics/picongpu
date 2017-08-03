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

#include "picongpu/simulation_defines.hpp"
#include <pmacc/nvidia/rng/RNG.hpp>
#include <pmacc/nvidia/rng/methods/Xor.hpp>
#include <pmacc/mpi/SeedPerRank.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

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
namespace detail
{
    /** call simple free user defined functor and provide a random number generator
     *
     * @tparam T_Distribution random number distribution
     * @tparam T_Seed seed to initialize the random number generator
     * @tparam T_SpeciesType type of the species that shall used to generate a unique
     *                       seed to initialize the random number generator
     */
    template<
        typename T_Distribution,
        typename T_Seed,
        typename T_SpeciesType
    >
    struct Rng
    {

        using Distribution = T_Distribution;
        using SpeciesType = T_SpeciesType;

        template< typename T_Acc >
        using RngType = pmacc::nvidia::rng::RNG<
            nvidia::rng::methods::Xor< T_Acc >,
            decltype( Distribution::get( std::declval< T_Acc >( ) ) )
        >;

        /** constructor
         *
         * @param currentStep current simulation time step
         */
        HINLINE Rng( uint32_t currentStep )
        {
            hostInit( currentStep );
        }

        /** create functor a random number generator
         *
         * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
         * @tparam T_Acc alpaka accelerator type
         *
         * @param alpaka accelerator
         * @param localSupercellOffset offset (in superCells, without any guards) relative
         *                        to the origin of the local domain
         * @param workerCfg configuration of the worker
         */
        template<
            typename T_WorkerCfg,
            typename T_Acc
        >
        DINLINE
        RngType< T_Acc > operator()(
            T_Acc const & acc,
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
            RngType< T_Acc > const rng = nvrng::create(
                nvidia::rng::methods::Xor< T_Acc >(
                    acc,
                    seed,
                    cellIdx
                ),
                Distribution::get( acc )
            );
            return rng;
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
                pmacc::traits::GetUniqueTypeId<
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

} // namespace detail
} // namepsace generic
} // namespace manipulators
} // namespace particles
} // namespace picongpu
