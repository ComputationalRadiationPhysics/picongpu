/* Copyright 2021-2023 Pawel Ordyna
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

#include "picongpu/particles/externalBeam/beam/Side.hpp"
#include "picongpu/particles/externalBeam/detail/StartAttributesContext.hpp"

#include <pmacc/random/RNGProvider.hpp>

#include <boost/mpl/apply.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace acc
            {
                //! Device side implementation of StartAttributes
                template<
                    typename T_Context,
                    typename T_AccStartPositionFunctor,
                    typename T_AccMomentumFunctor,
                    typename... T_AccFunctor>
                struct StartAttributes
                    : private T_AccStartPositionFunctor
                    , private T_AccMomentumFunctor
                    , private T_AccFunctor...
                {
                public:
                    DINLINE StartAttributes(
                        T_Context const& context,
                        T_AccStartPositionFunctor const&& accStartPositionFunctor,
                        T_AccMomentumFunctor const&& accMomentumFunctor,
                        T_AccFunctor const&&... accFunctor)
                        : T_AccStartPositionFunctor(accStartPositionFunctor)
                        , T_AccMomentumFunctor(accMomentumFunctor)
                        , T_AccFunctor(accFunctor)...
                        , context_m(context)
                    {
                    }
                    /** Set in-cell position, weighting, and extra attributes
                     *
                     * @tparam T_Worker lockstep worker type
                     * @tparam T_Particle pmacc::Particle, particle type
                     * @tparam T_Args pmacc::Particle, arbitrary number of particles types
                     *
                     * @param worker lockstep worker
                     * @param particle particle to be manipulated
                     * @param ... unused particles
                     */
                    template<typename T_Acc, typename T_Particle, typename... T_Args>
                    DINLINE void operator()(T_Acc const& worker, T_Particle& particle, T_Args&&...)
                    {
                        // set position and weighting
                        T_AccStartPositionFunctor::operator()(context_m, particle);
                        // set momentum (needs weighting to be already set)
                        T_AccMomentumFunctor::operator()(context_m, particle);
                        // execute additional functors  e.g. setting the startPhase attribute
                        (T_AccFunctor::operator()(context_m, particle), ...);
                    }


                    /** Get the number of macro particles that should be created and initialize this functor
                     *
                     * This is called only once before the operator is called. Hence this method is also used to
                     * initialize this functor and the  sub-functors. There is one instance of the low level functor
                     * for each cell in KernelFillGridWithParticles so that the initialization can depend on the cell
                     * position.
                     *
                     * @tparam T_Particle type of the particles that should be created
                     *
                     * @param realParticlesPerCell number of new real particles in the cell in which this instance
                     * creates particles
                     *
                     * @return number of macro particles that need to be created in this cell (The operator() will be
                     * called that many times)
                     */
                    template<typename T_Particle>
                    DINLINE uint32_t numberOfMacroParticles(float_X const realParticlesPerCell)
                    {
                        // numberOfMacroParticles also initializes the start position functor
                        const uint32_t numMacroParticles
                            = T_AccStartPositionFunctor::template numberOfMacroParticles<T_Particle>(
                                context_m,
                                realParticlesPerCell);

                        return numMacroParticles;
                    }

                private:
                    PMACC_ALIGN(context_m, T_Context);
                };
            } // namespace acc


            //! Host factory implementation for StartAttributes
            template<
                typename T_Species,
                typename T_StartPositionFunctor,
                typename T_MomentumFunctor,
                typename... T_Functor>
            struct StartAttributesImpl
                : private T_StartPositionFunctor
                , private T_MomentumFunctor
                , private T_Functor...

            {
                using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;

                HINLINE StartAttributesImpl(uint32_t const& currentStep)
                    : T_StartPositionFunctor(currentStep)
                    , T_MomentumFunctor(currentStep)
                    , T_Functor(currentStep)...
                    , rngHandle(RNGFactory::createHandle())
                {
                }

                /** Create functor for the accelerator
                 *
                 * @tparam T_Worker lockstep worker type
                 *
                 * @param worker lockstep worker
                 * @param localSupercellOffset offset (in superCells, without any guards) relative
                 *                        to the origin of the local domain
                 */
                template<typename T_Worker>
                DINLINE auto operator()(T_Worker const& worker, DataSpace<simDim> const& localSupercellOffset) const
                {
                    // initialize a random nuber generator for the sub-functors
                    RNGFactory::Handle rngHandleLocal = rngHandle;
                    rngHandleLocal.init(
                        localSupercellOffset * SuperCellSize::toRT()
                        + DataSpaceOperations<simDim>::template map<SuperCellSize>(worker.getWorkerIdx()));
                    // combine worker information and the number generator in a context variable
                    auto context{detail::makeContext(worker, rngHandleLocal)};

                    return acc::StartAttributes<
                        ALPAKA_DECAY_T(decltype(context)),
                        ALPAKA_DECAY_T(decltype(T_StartPositionFunctor::operator()(worker, localSupercellOffset))),
                        ALPAKA_DECAY_T(decltype(T_MomentumFunctor::operator()(worker, localSupercellOffset))),
                        ALPAKA_DECAY_T(decltype(T_Functor::operator()(worker, localSupercellOffset)))...>(
                        context,
                        T_StartPositionFunctor::operator()(worker, localSupercellOffset),
                        T_MomentumFunctor::operator()(worker, localSupercellOffset),
                        T_Functor::operator()(worker, localSupercellOffset)...);
                }

                static HINLINE std::string getName()
                {
                    return std::string("StartAttributes");
                }

            private:
                PMACC_ALIGN(rngHandle, RNGFactory::Handle);
            };


            /*
             * We need this wrapper since we can not use a placeholder for T_Species since we are using a variadic
             * template for the extra functors.
             */
            template<typename T_StartPositionFunctor, typename T_MomentumFunctor, typename... T_Functor>
            struct StartAttributes
                : private T_StartPositionFunctor
                , private T_MomentumFunctor
                , private T_Functor...

            {
                template<typename T_Species>
                struct apply
                {
                    using type = StartAttributesImpl<
                        T_Species,
                        typename boost::mpl::apply1<T_StartPositionFunctor, T_Species>::type,
                        typename boost::mpl::apply1<T_MomentumFunctor, T_Species>::type,
                        typename boost::mpl::apply1<T_Functor, T_Species>::type...>;
                };
            };
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
