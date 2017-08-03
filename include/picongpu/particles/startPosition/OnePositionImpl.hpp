/* Copyright 2013-2017 Axel Huebl, Heiko Burau, Rene Widera
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

#include <boost/mpl/integral_c.hpp>


namespace picongpu
{
namespace particles
{
namespace startPosition
{
namespace acc
{

    template< typename T_ParamClass >
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
        template<
            typename T_Particle,
            typename ... T_Args
        >
        DINLINE void operator()(
            T_Particle & particle,
            T_Args && ...
        )
        {
            particle[ position_ ] = T_ParamClass{}.inCellOffset.template shrink< simDim >( );
            particle[ weighting_ ] = m_weighting;
        }

        DINLINE uint32_t
        numberOfMacroParticles( float_X const realParticlesPerCell )
        {
            return detail::WeightMacroParticles{}(
                realParticlesPerCell,
                T_ParamClass::numParticlesPerCell,
                m_weighting
            );
        }

        float_X m_weighting;
    };

} // namespace acc
} // namespace startPosition
} // namespace particles
} // namespace picongpu
