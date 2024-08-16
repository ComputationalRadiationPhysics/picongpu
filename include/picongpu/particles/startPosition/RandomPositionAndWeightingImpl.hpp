/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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
                    /** Set in-cell position and weighting
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

                        // The last macroparticle of a cell gets the remaining weight
                        if(m_remainingMacroparticles <= 1)
                        {
                            particle[weighting_] = m_totalRemainingWeighting + MIN_WEIGHTING;
                            m_totalRemainingWeighting = 0.0_X;
                            m_remainingMacroparticles = 0;
                        }
                        else
                        {
                            /* Generate a weighting uniformly distributed in [0, 2x average weighting).
                             * This is a weighting on top of MIN_WEIGHTING.
                             * Clump it to be withing the remaining weighting for this cell.
                             */
                            auto weighting = rng() * 2.0_X * m_averageWeighting;
                            weighting = math::min(weighting, m_totalRemainingWeighting);
                            particle[weighting_] = weighting + MIN_WEIGHTING;
                            m_totalRemainingWeighting -= weighting;
                            m_remainingMacroparticles--;
                        }
                    }

                    /** Get the number of macroparticles for the current cell
                     *
                     * A user must call operator() for this object exactly as many times.
                     *
                     * @tparam T_Particle particle type
                     *
                     * @param realParticlesPerCell number of real particles for the cell
                     */
                    template<typename T_Particle>
                    HDINLINE uint32_t numberOfMacroParticles(float_X const realParticlesPerCell)
                    {
                        m_remainingMacroparticles = startPosition::detail::WeightMacroParticles{}(
                            realParticlesPerCell,
                            T_ParamClass::numParticlesPerCell,
                            m_averageWeighting);
                        m_averageWeighting = math::max(m_averageWeighting - MIN_WEIGHTING, 0.0_X);
                        m_totalRemainingWeighting = m_averageWeighting * m_remainingMacroparticles;
                        return m_remainingMacroparticles;
                    }

                    /** Average weighting of a macroparticle on top of MIN_WEIGHTING
                     *
                     * Due to the initialization logic, will always be >= 0
                     */
                    float_X m_averageWeighting;

                    //! Total weighting for remaining macroparticles on top of MIN_WEIGHTING for each particle
                    float_X m_totalRemainingWeighting;

                    //! Number of macroparticles remaining to be generated
                    uint32_t m_remainingMacroparticles;
                };

            } // namespace acc
        } // namespace startPosition
    } // namespace particles
} // namespace picongpu
