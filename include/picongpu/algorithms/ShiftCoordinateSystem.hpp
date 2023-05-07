/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/types.hpp>

namespace picongpu
{
    /** shift to new coordinate system
     *
     * @tparam T_supports support of the particle shape
     */
    template<uint32_t T_supports>
    struct ShiftCoordinateSystem
    {
        /** shift to new coordinate system
         *
         * shift DataBox and vector to new coordinate system
         * @param[in,out] dataBox DataBox pointing to the particle located cell
         * @param[in,out] pos position of the particle
         *                    - defined for [0.0;1.0) per dimension
         * @param fieldPos vector with relative coordinates for shift ( value range [0.0;0.5] )
         *
         * After this coordinate shift vector has well defined ranges per dimension,
         * for each defined fieldPos:
         *
         * - Even Support: vector is always [0.0;1.0)
         * - Odd Support: vector is always [-0.5;0.5)
         */
        template<typename T_DataBox, typename T_Vector, typename T_FieldType>
        HDINLINE void operator()(T_DataBox& dataBox, T_Vector& pos, const T_FieldType& fieldPos)
        {
            constexpr uint32_t dim = T_Vector::dim;
            using ValueType = typename T_Vector::type;
            constexpr uint32_t support = T_supports;
            constexpr bool isEven = (support % 2) == 0;

            DataSpace<dim> intShift;
            PMACC_UNROLL(dim)
            for(uint32_t d = 0; d < dim; ++d)
            {
                const ValueType v_pos = pos[d] - fieldPos[d] - ValueType{0.5};
                if constexpr(isEven)
                {
                    // pos range [-1.0;0.5)
                    intShift[d] = v_pos >= ValueType{-0.5} ? 0 : -1;
                }
                else
                {
                    // pos range [-1.0;0.5)
                    intShift[d] = v_pos >= ValueType{0.0} ? 1 : 0;
                }
                pos[d] = v_pos - ValueType(intShift[d]) + ValueType{0.5};
            }
            dataBox = dataBox.shift(intShift);
        }
    };

} // namespace picongpu
