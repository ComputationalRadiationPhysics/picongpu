/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/startPosition/detail/WeightMacroParticles.def"
#include "picongpu/particles/startPosition/generic/FreeRng.def"

namespace picongpu
{
    namespace particles
    {
        namespace startPosition
        {
            namespace acc
            {
                template<typename T_ParamClass>
                struct RandomImpl
                {
                    /** set in-cell position and weighting
                     *
                     * @tparam T_Rng functor::misc::RngWrapper, type of the random number generator
                     * @tparam T_Particle pmacc::Particle, particle type
                     *
                     * @param rng random number generator
                     * @param particle particle to be manipulated
                     */
                    template<typename T_Rng, typename T_Particle>
                    HDINLINE void operator()(T_Rng& rng, T_Particle& particle)
                    {
                        floatD_X tmpPos;

                        for(uint32_t d = 0; d < simDim; ++d)
                            tmpPos[d] = rng();

                        particle[position_] = tmpPos;
                        particle[weighting_] = m_weighting;
                    }

                    template<typename T_Particle>
                    HDINLINE uint32_t numberOfMacroParticles(float_X const realParticlesPerCell)
                    {
                        return startPosition::detail::WeightMacroParticles{}(
                            realParticlesPerCell,
                            T_ParamClass::numParticlesPerCell,
                            m_weighting);
                    }

                    float_X m_weighting;
                };

            } // namespace acc
        } // namespace startPosition
    } // namespace particles
} // namespace picongpu
