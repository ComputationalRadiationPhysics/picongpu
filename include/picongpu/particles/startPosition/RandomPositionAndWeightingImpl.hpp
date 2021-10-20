/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund, Sergei Bastrakov
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

#include "picongpu/particles/startPosition/detail/WeightMacroParticles.hpp"
#include "picongpu/particles/startPosition/generic/FreeRng.def"

#include <boost/mpl/integral_c.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace startPosition
        {
            namespace acc
            {
                template<typename T_ParamClass>
                struct RandomPositionAndWeightingImpl
                {
                    /** set in-cell position and weighting
                     *
                     * @tparam T_Rng functor::misc::RngWrapper, type of the random number generator
                     * @tparam T_Particle pmacc::Particle, particle type
                     * @tparam T_Args pmacc::Particle, arbitrary number of particles types
                     *
                     * @param rng random number generator
                     * @param particle particle to be manipulated
                     * @param ... unused particles
                     */
                    template<typename T_Rng, typename T_Particle, typename... T_Args>
                    HDINLINE void operator()(T_Rng& rng, T_Particle& particle, T_Args&&...)
                    {
                        floatD_X tmpPos;
                        for(uint32_t d = 0; d < simDim; ++d)
                            tmpPos[d] = rng();
                        particle[position_] = tmpPos;

                        // The last macroparticle of a cell gets the remaining weight
                        if(m_remainingMacroparticles <= 1)
                        {
                            particle[weighting_] = math::max(m_totalRemainingWeighting, MIN_WEIGHTING);
                            m_totalRemainingWeighting = 0.0_X;
                            m_remainingMacroparticles = 0;
                        }
                        else
                        {
                            m_remainingMacroparticles--;
                            // Generate a weighting uniformly distributed in [0, 2x average weighting]
                            auto weighting = rng() * 2.0_X * m_averageWeighting;
                            /* Clump it to the valid range: it has to be at least MIN_WEIGHTING and
                             * all remaining macroparticles also have at least MIN_WEIGHTING
                             */
                            auto const maxWeighting = m_totalRemainingWeighting
                                - MIN_WEIGHTING * static_cast<float_X>(m_remainingMacroparticles);
                            weighting = math::max(math::min(weighting, maxWeighting), MIN_WEIGHTING);
                            particle[weighting_] = weighting;
                            m_totalRemainingWeighting -= weighting;
                        }
                    }

                    template<typename T_Particle>
                    HDINLINE uint32_t numberOfMacroParticles(float_X const realParticlesPerCell)
                    {
                        m_remainingMacroparticles = startPosition::detail::WeightMacroParticles{}(
                            realParticlesPerCell,
                            T_ParamClass::numParticlesPerCell,
                            m_averageWeighting);
                        m_totalRemainingWeighting = m_averageWeighting * m_remainingMacroparticles;
                        return m_remainingMacroparticles;
                    }

                    /** Average weighting of a macroparticle
                     *
                     * Due to the initialization logic, will always be >= MIN_WEIGHTING
                     */
                    float_X m_averageWeighting;

                    //! Total weighting for remaining macroparticles
                    float_X m_totalRemainingWeighting;

                    //! Number of macroparticles remaining to be generated
                    uint32_t m_remainingMacroparticles;
                };

            } // namespace acc
        } // namespace startPosition
    } // namespace particles
} // namespace picongpu
