/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
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
#include "picongpu/particles/startPosition/OnePositionImpl.def"
#include "picongpu/particles/startPosition/detail/WeightMacroParticles.hpp"

#include <pmacc/traits/HasIdentifier.hpp>

#include <boost/mpl/integral_c.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace startPosition
        {
            namespace acc
            {
                namespace detail
                {
                    template<bool T_hasWeighting>
                    struct SetWeighting
                    {
                        template<typename T_Particle>
                        HDINLINE void operator()(T_Particle& particle, float_X const weighting)
                        {
                            particle[weighting_] = weighting;
                        }
                    };

                    template<>
                    struct SetWeighting<false>
                    {
                        template<typename T_Particle>
                        HDINLINE void operator()(T_Particle&, float_X const)
                        {
                        }
                    };

                } // namespace detail

                template<typename T_ParamClass>
                struct OnePositionImpl
                {
                    /** set in-cell position and weighting
                     *
                     * @tparam T_Particle pmacc::Particle, particle type
                     * @tparam T_Args pmacc::Particle, arbitrary number of particles types
                     *
                     * @param particle particle to be manipulated
                     * @param ... unused particles
                     */
                    template<typename T_Particle, typename... T_Args>
                    HDINLINE void operator()(T_Particle& particle, T_Args&&...)
                    {
                        particle[position_] = T_ParamClass{}.inCellOffset.template shrink<simDim>();

                        // set the weighting attribute if the particle species has it
                        bool const hasWeighting
                            = pmacc::traits::HasIdentifier<typename T_Particle::FrameType, weighting>::type::value;
                        detail::SetWeighting<hasWeighting> setWeighting;
                        setWeighting(particle, m_weighting);
                    }

                    template<typename T_Particle>
                    HDINLINE uint32_t numberOfMacroParticles(float_X const realParticlesPerCell)
                    {
                        bool const hasWeighting
                            = pmacc::traits::HasIdentifier<typename T_Particle::FrameType, weighting>::type::value;

                        // note: m_weighting member might stay uninitialized!
                        uint32_t result(T_ParamClass::numParticlesPerCell);

                        if(hasWeighting)
                            result = startPosition::detail::WeightMacroParticles{}(
                                realParticlesPerCell,
                                T_ParamClass::numParticlesPerCell,
                                m_weighting);

                        return result;
                    }

                private:
                    float_X m_weighting;
                };

            } // namespace acc
        } // namespace startPosition
    } // namespace particles
} // namespace picongpu
