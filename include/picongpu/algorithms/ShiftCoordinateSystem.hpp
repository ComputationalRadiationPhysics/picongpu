/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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
#include <pmacc/math/vector/Int.hpp>

#include <boost/mpl/range_c.hpp>
#include <boost/mpl/vector.hpp>
#include <pmacc/meta/AllCombinations.hpp>
#include <pmacc/meta/ForEach.hpp>

namespace picongpu
{
    /** calculate offset to move coordinate system in an easy to use system
     *
     * There are two cases:
     *  - system with even shape and odd shape
     *  - for more see documentation of the implementation
     */
    template<bool T_isEvenShape>
    struct GetOffsetToStaticShapeSystem;

    template<typename T_Component, typename T_Supports>
    struct AssignToDim
    {
        template<typename T_Type, typename T_Vector, typename T_FieldType>
        HDINLINE void operator()(T_Type& cursor, T_Vector& pos, const T_FieldType& fieldPos)
        {
            const uint32_t dim = T_Vector::dim;
            using ValueType = typename T_Vector::type;

            using Supports = T_Supports;
            using Component = T_Component;

            const uint32_t component = Component::x::value;
            const uint32_t support = Supports::template at<component>::type::value;
            const bool isEven = (support % 2) == 0;


            const ValueType v_pos = pos[component] - fieldPos[component];
            DataSpace<dim> intShift;
            intShift[component] = GetOffsetToStaticShapeSystem<isEven>()(v_pos);
            cursor = cursor(intShift);
            pos[component] = v_pos - ValueType(intShift[component]);
        }
    };

    /** shift to new coordinate system
     *
     * @tparam T_supports CT::Vector with support
     */
    template<typename T_supports>
    struct ShiftCoordinateSystem
    {
        /** shift to new coordinate system
         *
         * shift cursor and vector to new coordinate system
         * @param[in,out] cursor cursor to memory
         * @param[in,out] vector short vector with coordinates in old system
         *                        - defined for [0.0;1.0) per dimension
         * @param fieldPos vector with relative coordinates for shift ( value range [0.0;0.5] )
         *
         * After this coordinate shift vector has well defined ranges per dimension,
         * for each defined fieldPos:
         *
         * - Even Support: vector is always [0.0;1.0)
         * - Odd Support: vector is always [-0.5;0.5)
         */
        template<typename T_Cursor, typename T_Vector, typename T_FieldType>
        HDINLINE void operator()(T_Cursor& cursor, T_Vector& vector, const T_FieldType& fieldPos)
        {
            /** \todo check if a static assert on
             *  "T_Cursor::dim" == T_Vector::dim ==  T_FieldType::dim is possible
             *  and does not waste registers */
            const uint32_t dim = T_Vector::dim;

            using Size = boost::mpl::vector1<boost::mpl::range_c<uint32_t, 0, dim>>;
            using CombiTypes = typename AllCombinations<Size>::type;

            meta::ForEach<CombiTypes, AssignToDim<bmpl::_1, T_supports>> shift;
            shift(cursor, vector, fieldPos);
        }
    };


    /** Offset calculation for even support
     *
     * @param pos position of the particle relative to the grid
     *            - defined for [-0.5;1.0)
     * @return offset for the old system ( new system = old_system - offset)
     */
    template<>
    struct GetOffsetToStaticShapeSystem<true>
    {
        template<typename T_Type>
        HDINLINE int operator()(const T_Type& pos)
        {
            return pmacc::math::float2int_rd(pos);
        }
    };


    /** Offset calculation for odd support
     *
     * @param pos position of the particle relative to the grid
     *            - defined for [-0.5;1.0)
     * @return offset for the old system ( new system = old_system - offset)
     */
    template<>
    struct GetOffsetToStaticShapeSystem<false>
    {
        template<typename T_Type>
        HDINLINE int operator()(const T_Type& pos)
        {
            return pos >= T_Type(0.5) ? 1 : 0;
        }
    };

} // namespace picongpu
