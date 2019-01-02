/* Copyright 2013-2019 Heiko Burau, Rene Widera, Axel Huebl
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
#include "picongpu/algorithms/DifferenceToUpper.def"

#include <pmacc/math/Vector.hpp>


namespace picongpu
{

    template< uint32_t T_dim >
    struct DifferenceToUpper
    {
        static constexpr uint32_t dim = T_dim;

        using OffsetOrigin = typename pmacc::math::CT::make_Int<
            dim,
            0
        >::type;
        using OffsetEnd = typename pmacc::math::CT::make_Int<
            dim,
            1
        >::type;

        /** calculate the difference for a given direction
         *
         * @tparam T_direction direction for the difference operation
         * @tparam T_isLesserThanDim not needed/ this is calculated by the compiler
         */
        template<
            uint32_t T_direction,
            bool T_isLesserThanDim = ( T_direction < dim )
        >
        struct GetDifference
        {
            static constexpr uint32_t direction = T_direction;

            HDINLINE GetDifference( )
            {
            }

            /** get difference to lower value
             * @return difference divided by cell size of the given direction
             */
            template< typename Memory >
            HDINLINE typename Memory::ValueType operator()( Memory const & mem ) const
            {
                // defaults to (0, 0, 0) in 3D
                DataSpace< dim > const indexIdentity;
                // e.g., (0, 1, 0) for d/dy in 3D
                DataSpace< dim > indexUpper;
                indexUpper[ direction ] = 1;

                return ( mem( indexUpper ) - mem( indexIdentity ) ) /
                    cellSize[ direction ];
            }
        };

        /** special case for `direction >= simulation dimensions`
         *
         *  difference = d/dx = 0
         */
        template< uint32_t T_direction >
        struct GetDifference<
            T_direction,
            false
        >
        {
            HDINLINE GetDifference( )
            {
            }

            /** @return always a zeroed value
             */
            template< typename Memory >
            HDINLINE typename Memory::ValueType operator()( Memory const & mem) const
            {
                return Memory::ValueType::create( 0.0 );
            }
        };

    };

} // namespace picongpu
