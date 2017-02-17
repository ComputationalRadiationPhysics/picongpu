/* Copyright 2017 Rene Widera
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

#include "particles/manipulators/FreeImpl.hpp"


namespace picongpu
{
namespace particles
{
namespace manipulators
{

    namespace detail
    {
        template<
            typename T_DestAttribute,
            typename T_SrcAttribute
        >
        struct CopyAttributeFunctor
        {
            /** copy attribute
             *
             * @tparam T_Particle particle type
             * @param particle particle to be manipulated
             */
            template< typename T_Particle >
            DINLINE void operator()( T_Particle const & particle )
            {
                particle[ T_DestAttribute{} ] = particle[ T_SrcAttribute{} ];
            }
        };
    } // namespace detail

} //namespace manipulators
} //namespace particles
} //namespace picongpu
