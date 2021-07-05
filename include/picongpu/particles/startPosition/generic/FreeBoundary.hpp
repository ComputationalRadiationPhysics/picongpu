/* Copyright 2013-2021 Rene Widera, Axel Huebl
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is FreeBoundary software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the FreeBoundary Software Foundation, either version 3 of the License, or
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

#include "picongpu/particles/functor/misc/DomainInfo.hpp"
#include "picongpu/particles/startPosition/generic/FreeBoundary.def"

#include <pmacc/random/RNGProvider.hpp>

#include <type_traits>
#include <utility>

namespace picongpu
{
    namespace particles
    {
        namespace startPosition
        {
            namespace generic
            {
                namespace acc
                {
                    /** execute the user functor
                     *
                     * @tparam T_Args type of the arguments passed to the user functor
                     * @tparam T_Acc alpaka accelerator type
                     *
                     * @param alpaka accelerator
                     * @param args arguments passed to the user functor
                     */
                    template<typename T_Functor, typename T_RngHandle>
                    template<typename... T_Args, typename T_Acc>
                    HDINLINE void FreeBoundary<T_Functor, T_RngHandle>::operator()(T_Acc const& acc, T_Args&&... args)
                    {
                        if(isBoundary())
                            Functor::operator()(acc, *this, args...);
                    }

                    template<typename T_Functor, typename T_RngHandle>
                    template<typename T_Particle>
                    HDINLINE uint32_t
                    FreeBoundary<T_Functor, T_RngHandle>::numberOfMacroParticles(float_X const realParticlesPerCell)
                    {
                        if(isBoundary())
                            return Functor::template numberOfMacroParticles<T_Particle>(*this, realParticlesPerCell);

                        return 0;
                    }

                } // namespace acc

                template<typename T_Functor>
                struct FreeBoundary : protected T_Functor
                {
                    using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
                    using RngHandle = typename RNGFactory::Handle;

                    functor::misc::DomainInfo domInfo;
                    RngHandle rngHandle;
                    uint32_t m_currentStep;

                    using Functor = T_Functor;

                    template<typename T_SpeciesType>
                    struct apply
                    {
                        using type = FreeBoundary;
                    };

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
                    template<typename DeferFunctor = Functor>
                    HINLINE FreeBoundary(
                        uint32_t currentStep,
                        typename std::enable_if<std::is_constructible<DeferFunctor, uint32_t>::value>::type* = 0)
                        : Functor(currentStep)
                        , domInfo(currentStep)
                        , m_currentStep(currentStep)
                        , rngHandle(RNGFactory::createHandle())
                    {
                    }

                    /** constructor
                     *
                     * This constructor is only compiled if the user functor has a default constructor.
                     *
                     * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
                     *                      the constructor
                     * @param current simulation time step
                     * @param is used to enable/disable the constructor (do not pass any value to this parameter)
                     */
                    template<typename DeferFunctor = Functor>
                    HINLINE FreeBoundary(
                        uint32_t currentStep,
                        typename std::enable_if<std::is_constructible<DeferFunctor>::value>::type* = 0)
                        : Functor()
                        , domInfo(currentStep)
                        , m_currentStep(currentStep)
                        , rngHandle(RNGFactory::createHandle())
                    {
                    }

                    /** create device functor
                     *
                     * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
                     * @tparam T_Acc alpaka accelerator type
                     *
                     * @param alpaka accelerator
                     * @param offset (in supercells, without any guards) to the
                     *         origin of the local domain
                     * @param configuration of the worker
                     */
                    template<typename T_WorkerCfg, typename T_Acc>
                    HDINLINE acc::FreeBoundary<Functor, RngHandle> operator()(
                        T_Acc const& acc,
                        DataSpace<simDim> const& localSupercellOffset,
                        T_WorkerCfg const& workerCfg) const
                    {
                        auto dom = domInfo(acc, localSupercellOffset, workerCfg);
                        RngHandle rng(rngHandle);
                        auto rngStateIdx = localSupercellOffset * SuperCellSize::toRT()
                            + DataSpaceOperations<simDim>::template map<SuperCellSize>(workerCfg.getWorkerIdx());
                        rng.init(rngStateIdx);
                        return acc::FreeBoundary<Functor, RngHandle>(
                            *static_cast<Functor const*>(this),
                            dom,
                            rng,
                            m_currentStep);
                    }
                };

            } // namespace generic
        } // namespace startPosition
    } // namespace particles
} // namespace picongpu
