/* Copyright 2015-2021 Rene Widera, Alexander Grund
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
#include "picongpu/particles/functor/misc/RngWrapper.hpp"

#include <pmacc/mpi/SeedPerRank.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>
#include <pmacc/random/methods/methods.hpp>
#include <pmacc/random/RNGProvider.hpp>

#include <utility>
#include <type_traits>
#include <string>


namespace picongpu
{
    namespace particles
    {
        namespace functor
        {
            namespace misc
            {
                /** call simple free user defined functor and provide a random number generator
                 *
                 * @tparam T_Distribution random number distribution
                 */
                template<typename T_Distribution>
                struct Rng
                {
                    using Distribution = T_Distribution;
                    using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
                    using RngHandle = typename RNGFactory::Handle;
                    using RandomGen = RngWrapper<cupla::Acc, typename RngHandle::GetRandomType<Distribution>::type>;

                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE Rng(uint32_t currentStep) : rngHandle(RNGFactory::createHandle())
                    {
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
                    template<typename T_WorkerCfg, typename T_Acc>
                    HDINLINE RandomGen operator()(
                        T_Acc const& acc,
                        DataSpace<simDim> const& localSupercellOffset,
                        T_WorkerCfg const& workerCfg) const
                    {
                        RngHandle tmp(rngHandle);
                        tmp.init(
                            localSupercellOffset * SuperCellSize::toRT()
                            + DataSpaceOperations<simDim>::template map<SuperCellSize>(workerCfg.getWorkerIdx()));
                        return RandomGen(acc, tmp.applyDistribution<Distribution>());
                    }

                private:
                    RngHandle rngHandle;
                };

            } // namespace misc
        } // namespace functor
    } // namespace particles
} // namespace picongpu
