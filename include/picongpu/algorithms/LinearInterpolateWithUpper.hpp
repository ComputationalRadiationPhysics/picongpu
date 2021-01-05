/* Copyright 2015-2021 Heiko Burau, Rene Widera
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
    /** Calculate linear interpolation to upper cell value
     *
     * @tparam T_Dim for how many dimensions does this operator interpolate
     *
     * If `GetDifference` is called for a direction greater or equal T_Dim,
     * a zeroed value is returned (assumes symmetry in those directions).
     */
    template<uint32_t T_Dim>
    struct LinearInterpolateWithUpper
    {
        static constexpr uint32_t dim = T_Dim;

        using OffsetOrigin = typename pmacc::math::CT::make_Int<dim, 0>::type;
        using OffsetEnd = typename pmacc::math::CT::make_Int<dim, 1>::type;

        /** calculate the linear interpolation for a given direction
         *
         * @tparam T_direction direction for the interpolation operation
         * @tparam T_isLesserThanDim not needed/ this is calculated by the compiler
         */
        template<uint32_t T_direction, bool T_isLesserThanDim = (T_direction < dim)>
        struct GetInterpolatedValue
        {
            static constexpr uint32_t direction = T_direction;

            /** get interpolated value
             * @return interpolated value
             */
            template<class Memory>
            HDINLINE typename Memory::ValueType operator()(const Memory& mem) const
            {
                const DataSpace<dim> indexIdentity; /* defaults to (0, 0, 0) in 3D */
                DataSpace<dim> indexUpper; /* e.g., (0, 1, 0) for direction y in 3D */
                indexUpper[direction] = 1;

                return (mem(indexUpper) + mem(indexIdentity)) * Memory::ValueType::create(0.5);
            }
        };

        /** special case for `direction >= simulation dimensions`*/
        template<uint32_t T_direction>
        struct GetInterpolatedValue<T_direction, false>
        {
            /** @return always identity
             */
            template<class Memory>
            HDINLINE typename Memory::ValueType operator()(const Memory& mem) const
            {
                return *mem;
            }
        };
    };

} // namespace picongpu
