/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Pawel Ordyna
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/startPosition/generic/Free.def"

namespace picongpu
{
    namespace particles
    {
        namespace startPosition
        {
            namespace acc
            {
                template<typename T_ParamClass>
                struct QuietImpl
                {
                    /** set in-cell position and weighting
                     *
                     * @warning It is not allowed to call this functor as many times as
                     *          the resulting value of numberOfMacroParticles.
                     *
                     * @tparam T_Particle pmacc::Particle, particle type
                     *
                     * @param particle particle to be manipulated
                     */
                    template<typename T_Particle>
                    HDINLINE void operator()(T_Particle& particle)
                    {
                        uint32_t maxNumMacroParticles
                            = pmacc::math::CT::volume<typename T_ParamClass::numParticlesPerDimension>::type::value;

                        /* reset the particle position if the operator is called more times
                         * than allowed (m_currentMacroParticles underflow protection for)
                         */
                        if(maxNumMacroParticles <= m_currentMacroParticles)
                            m_currentMacroParticles = maxNumMacroParticles - 1u;

                        // spacing between particles in each direction in the cell
                        floatD_X spacing;
                        for(uint32_t i = 0; i < simDim; ++i)
                            spacing[i] = float_X(1.0) / float_X(m_numParDirection[i]);

                        /* coordinate in the local in-cell lattice
                         *   x = [0, numParsPerCell_X-1]
                         *   y = [0, numParsPerCell_Y-1]
                         *   z = [0, numParsPerCell_Z-1]
                         */
                        DataSpace<simDim> inCellCoordinate
                            = pmacc::math::mapToND(m_numParDirection, static_cast<int>(m_currentMacroParticles));

                        particle[position_]
                            = precisionCast<float_X>(inCellCoordinate) * spacing + spacing * float_X(0.5);
                        particle[weighting_] = m_weighting;

                        --m_currentMacroParticles;
                    }

                    template<typename T_Particle>
                    HDINLINE uint32_t numberOfMacroParticles(float_X const realParticlesPerCell)
                    {
                        m_numParDirection = T_ParamClass::numParticlesPerDimension::toRT();

                        m_weighting = float_X(0.0);
                        uint32_t numMacroParticles
                            = pmacc::math::CT::volume<typename T_ParamClass::numParticlesPerDimension>::type::value;

                        if(numMacroParticles > 0u)
                            m_weighting = realParticlesPerCell / float_X(numMacroParticles);

                        while(m_weighting < MIN_WEIGHTING && numMacroParticles > 0u)
                        {
                            /* decrement component with greatest value*/
                            uint32_t max_component = 0u;
                            for(uint32_t i = 1; i < simDim; ++i)
                            {
                                if(m_numParDirection[i] > m_numParDirection[max_component])
                                    max_component = i;
                            }
                            m_numParDirection[max_component] -= 1u;

                            numMacroParticles = m_numParDirection.productOfComponents();

                            if(numMacroParticles > 0u)
                                m_weighting = realParticlesPerCell / float_X(numMacroParticles);
                            else
                                m_weighting = float_X(0.0);
                        }
                        m_currentMacroParticles = numMacroParticles - 1u;
                        return numMacroParticles;
                    }

                private:
                    PMACC_ALIGN(m_numParDirection, DataSpace<simDim>);
                    float_X m_weighting;
                    uint32_t m_currentMacroParticles;
                };

            } // namespace acc
        } // namespace startPosition
    } // namespace particles
} // namespace picongpu
