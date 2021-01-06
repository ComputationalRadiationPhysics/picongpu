/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund
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
#include "picongpu/particles/startPosition/generic/FreeRng.def"
#include "picongpu/particles/startPosition/detail/WeightMacroParticles.hpp"

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
                struct RandomImpl
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
