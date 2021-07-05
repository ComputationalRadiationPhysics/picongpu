/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Pawel Ordyna
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

#include "picongpu/plugins/externalBeam/AxisSwap.hpp"
#include "picongpu/plugins/externalBeam/Side.hpp"

#include <boost/mpl/integral_c.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace acc
            {
                template<typename T_StartPositionFunctor, typename T_MomentumFunctor, typename T_PhaseFunctor>
                struct StartAttributesImpl
                {
                    using StartPositionFunctor = T_StartPositionFunctor;
                    using MomentumFunctor = T_MomentumFunctor;
                    using PhaseFunctor = T_PhaseFunctor;

                public:
                    HINLINE StartAttributesImpl(uint32_t currentStep)
                        : startPositionFunctor(StartPositionFunctor(currentStep))
                        , momentumFunctor()
                        , phaseFunctor()
                    {
                    }
                    /** set in-cell position, weighting, momentum, phase
                     *
                     * @warning It is not allowed to call this functor as many times as
                     *          the resulting value of numberOfMacroParticles.
                     *
                     * @tparam T_Acc alpaka accelerator type
                     * @tparam T_MetaData a generic functor type which inherits from this class and calls this functor
                     * @tparam T_Particle pmacc::Particle, particle type
                     * @tparam T_Args pmacc::Particle, arbitrary number of particles types
                     *
                     * @param acc alpaka accelerator
                     * @param meta the instance of T_MetaData that calls this functor
                     * @param particle particle to be manipulated
                     * @param ... unused particles
                     */
                    template<typename T_Acc, typename T_MetaData, typename T_Particle, typename... T_Args>
                    HDINLINE void operator()(T_Acc const& acc, T_MetaData& meta, T_Particle& particle, T_Args&&...)
                    {
                        // set position and weighting
                        startPositionFunctor(acc, meta, particle);
                        // set momentum (needs weighting to be already set)
                        momentumFunctor(acc, meta, particle);
                        // set phase (needs position already set and may need momentum already set)
                        // this does nothing if T_PhaseFunctor = particles::externalBeam::phase::NoPhase .
                        phaseFunctor(acc, meta, particle);
                    }

                    template<typename T_Particle, typename T_MetaData>
                    HDINLINE uint32_t
                    numberOfMacroParticles(T_MetaData const& meta, float_X const realParticlesPerCell)
                    {
                        momentumFunctor.template init<T_Particle>(meta);
                        phaseFunctor.template init<T_Particle>(meta);
                        // numberOfMacroParticles also initializes a start position functor
                        const uint32_t numMacroParticles
                            = startPositionFunctor.template numberOfMacroParticles<T_Particle>(
                                meta,
                                realParticlesPerCell);

                        return numMacroParticles;
                    }

                private:
                    PMACC_ALIGN(startPositionFunctor, StartPositionFunctor);
                    PMACC_ALIGN(momentumFunctor, MomentumFunctor);
                    PMACC_ALIGN(phaseFunctor, PhaseFunctor);
                };
            } // namespace acc
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
