/*
 * SPDX-FileCopyrightText: Rene Widera, Finn-Ole Carstens
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/traits/HasIdentifier.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace transitionRadiation
        {
            /** read the `transitionRadiationMask` of a species */
            template<bool hasTransitionRadiationMask>
            struct GetTransitionRadiationMask
            {
                /** get the attribute value of `transitionRadiationMask`
                 *
                 * @param particle particle to be used
                 * @return value of the attribute `transitionRadiationMask`
                 */
                template<typename T_Particle>
                HDINLINE bool operator()(T_Particle const& particle) const
                {
                    return particle[transitionRadiationMask_];
                }
            };

            /** specialization
             *
             * specialization for the case that the species not owns the attribute
             * `transitionRadiationMask`
             */
            template<>
            struct GetTransitionRadiationMask<false>
            {
                /** get the attribute value of `transitionRadiationMask`
                 *
                 * @param particle to be used
                 * @return always true
                 */
                template<typename T_Particle>
                HDINLINE bool operator()(T_Particle const&) const
                {
                    return true;
                }
            };

            /** get the value of the particle attribute `transitionRadiationMask`
             *
             * Allow to read out the value of the attribute `transitionRadiationMask` also if
             * it is not defined for the particle.
             *
             * @tparam T_Particle particle type
             * @param particle valid particle
             * @return particle attribute value `transitionRadiationMask`, always `true` if attribute
             * `transitionRadiationMask` is not defined
             */
            template<typename T_Particle>
            HDINLINE bool getTransitionRadiationMask(T_Particle const& particle)
            {
                constexpr bool hasTransitionRadiationMask = pmacc::traits::
                    HasIdentifier<typename T_Particle::FrameType, transitionRadiationMask>::type::value;
                return GetTransitionRadiationMask<hasTransitionRadiationMask>{}(particle);
            }

        } // namespace transitionRadiation
    } // namespace plugins
} // namespace picongpu
