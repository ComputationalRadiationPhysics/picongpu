/* Copyright 2013-2020 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Klaus Steiniger, Felix Schmitt, Benjamin Worpitz,
 *                     Juncheng E, Sergei Bastrakov
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

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>

#include <cstdint>


namespace picongpu
{
namespace plugins
{
namespace xrayDiffraction
{
namespace detail
{

    /** Parameters of the reciprocal space
     * 
     * They define the regular 3d Cartesian lattice of scattering vectors q.
     * The scattering vector here is defined as 4 * pi * sin(theta) / lambda,
     * where 2 * theta is the angle between scattered and incident beam.
     */
    struct ReciprocalSpace
    {
        //! Lattice start
        float3_X min;

        //! Lattice step
        float3_X step;

        //! Number of points per each direction
        pmacc::DataSpace< 3 > size;

        /** Create a reciprocal space
         *
         * @param min lattice start
         * @param step lattice step
         * @param size number of points per each direction
         */
        /// TODO: figure out why 0._X instead of 0.0 on Hemera gives
        /// error #2486: user-defined literal operator not found
        HDINLINE ReciprocalSpace(
            float3_X const & min = float3_X::create( 0.0 ),
            float3_X const & step = float3_X::create( 0.0 ),
            pmacc::DataSpace< 3 > const & size = pmacc::DataSpace< 3 >::create( 1 )
        );

        /** Get scattering vector value by linear index
         *
         * @param linearIdx linear index within the product of size components
         */
        HDINLINE float3_X getValue( uint32_t const linearIdx ) const;

    };

    ReciprocalSpace::ReciprocalSpace(
        float3_X const & min,
        float3_X const & step,
        pmacc::DataSpace< 3 > const & size
    ):
        min( min ),
        step( step ),
        size( size )
    {
    }

    float3_X ReciprocalSpace::getValue( uint32_t const linearIdx ) const
    {
        pmacc::DataSpace< 3 > idx;
        idx[ 2 ] = linearIdx % size.z();
        idx[ 1 ] = ( linearIdx / size.z() ) % size.y();
        idx[ 0 ] = linearIdx / ( size.z() * size.y() );
        return min + step * precisionCast< float_X >( idx );
    }

} // namespace detail
} // namespace xrayDiffraction
} // namespace plugins
} // namespace picongpu
