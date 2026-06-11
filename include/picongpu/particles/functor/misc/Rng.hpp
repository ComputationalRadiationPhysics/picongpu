/*
 * SPDX-FileCopyrightText: Rene Widera, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/functor/misc/RngWrapper.hpp"

#include <pmacc/mpi/SeedPerRank.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/methods/methods.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

#include <string>
#include <type_traits>
#include <utility>

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

                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE Rng(uint32_t currentStep) : rngHandle(RNGFactory::createHandle())
                    {
                    }

                    /** create functor a random number generator
                     *
                     * @tparam T_Worker lockstep::Worker, lockstep worker type
                     *
                     * @param worker lockstep worker
                     * @param localSupercellOffset offset (in superCells, without any guards) relative
                     *                        to the origin of the local domainrker
                     */
                    template<typename T_Worker>
                    HDINLINE auto operator()(T_Worker const& worker, DataSpace<simDim> const& localSupercellOffset)
                        const
                    {
                        RngHandle tmp(rngHandle);
                        auto rngOffset = DataSpace<simDim>::create(0);
                        rngOffset.x() = worker.workerIdx();
                        auto numRNGsPerSuperCell = DataSpace<simDim>::create(1);
                        numRNGsPerSuperCell.x() = numFrameSlots;
                        tmp.init(localSupercellOffset * numRNGsPerSuperCell + rngOffset);
                        using RandomGen = RngWrapper<T_Worker, typename RngHandle::GetRandomType<Distribution>::type>;
                        return RandomGen(worker, tmp.applyDistribution<Distribution>());
                    }

                private:
                    RngHandle rngHandle;
                };

            } // namespace misc
        } // namespace functor
    } // namespace particles
} // namespace picongpu
