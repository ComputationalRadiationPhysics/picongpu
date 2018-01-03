/* Copyright 2013-2018 Heiko Burau, Rene Widera, Axel Huebl
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

#include <pmacc/types.hpp>
#include <pmacc/math/Vector.hpp>


namespace picongpu
{

/** calculate difference to upper value
 *
 * @tparam T_Dim for how many dimensions this operator access memory
 *
 * If `GetDifference` is called for a direction greater equal T_Dim always
 * a zeroed value is returned
 */
template<uint32_t T_Dim>
struct DifferenceToUpper
{
    static constexpr uint32_t dim = T_Dim;

    typedef typename pmacc::math::CT::make_Int<dim, 0>::type OffsetOrigin;
    typedef typename pmacc::math::CT::make_Int<dim, 1>::type OffsetEnd;

    /** calculate the difference for a given direction
     *
     * @tparam T_direction direction for the difference operation
     * @tparam T_isLesserThanDim not needed/ this is calculated by the compiler
     */
    template<uint32_t T_direction, bool T_isLesserThanDim = (T_direction < dim)>
    struct GetDifference
    {
        static constexpr uint32_t direction = T_direction;

        HDINLINE GetDifference()
        {
        }

        /** get difference to lower value
         * @return difference divided by cell size of the given direction
         */
        template<class Memory >
        HDINLINE typename Memory::ValueType operator()(const Memory& mem) const
        {
            const DataSpace<dim> indexIdentity; /* defaults to (0, 0, 0) in 3D */
            DataSpace<dim> indexUpper; /* e.g., (0, 1, 0) for d/dy in 3D */
            indexUpper[direction] = 1;

            return ( mem(indexUpper) - mem(indexIdentity)) / cellSize[direction];
        }
    };

    /** special case for `direction >= simulation dimensions`
     *
     *  difference = d/dx = 0
     */
    template<uint32_t T_direction>
    struct GetDifference<T_direction, false>
    {
        HDINLINE GetDifference()
        {
        }

        /** @return always a zeroed value
         */
        template<class Memory >
        HDINLINE typename Memory::ValueType operator()(const Memory& mem) const
        {
            return Memory::ValueType::create(0.0);
        }
    };

};

} //namespace picongpu
