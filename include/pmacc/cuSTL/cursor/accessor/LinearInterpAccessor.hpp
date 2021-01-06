/* Copyright 2015-2021 Heiko Burau
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/cuSTL/cursor/traits.hpp"
#include "pmacc/traits/GetInitializedInstance.hpp"
#include "pmacc/algorithms/math/defines/modf.hpp"
#include "pmacc/math/vector/Int.hpp"

namespace pmacc
{
    namespace cursor
    {
        /** Performs a 1D, 2D or 3D, linear interpolation on access.
         *
         * \tparam T_Cursor input data
         */
        template<typename T_Cursor, int dim = cursor::traits::dim<T_Cursor>::value>
        struct LinearInterpAccessor;

        template<typename T_Cursor>
        struct LinearInterpAccessor<T_Cursor, DIM1>
        {
            typedef T_Cursor Cursor;
            typedef typename Cursor::ValueType type;

            Cursor cursor;

            /**
             * @param cursor input data
             */
            HDINLINE LinearInterpAccessor(const Cursor& cursor) : cursor(cursor)
            {
            }

            template<typename T_Position>
            HDINLINE type operator()(const T_Position pos) const
            {
                BOOST_STATIC_ASSERT(T_Position::dim == DIM1);

                T_Position intPart;
                T_Position fracPart;

                fracPart[0] = pmacc::math::modf(pos[0], &(intPart[0]));

                const math::Int<DIM1> idx1D(static_cast<int>(intPart[0]));

                type result = pmacc::traits::GetInitializedInstance<type>()(0.0);
                typedef typename T_Position::type PositionComp;
                for(int i = 0; i < 2; i++)
                {
                    const PositionComp weighting1D = (i == 0 ? (PositionComp(1.0) - fracPart[0]) : fracPart[0]);
                    result += static_cast<type>(weighting1D * this->cursor[idx1D + math::Int<DIM1>(i)]);
                }

                return result;
            }
        };

        template<typename T_Cursor>
        struct LinearInterpAccessor<T_Cursor, DIM2>
        {
            typedef T_Cursor Cursor;
            typedef typename T_Cursor::ValueType type;

            Cursor cursor;

            /**
             * @param cursor input data
             */
            HDINLINE LinearInterpAccessor(const Cursor& cursor) : cursor(cursor)
            {
            }

            template<typename T_Position>
            HDINLINE type operator()(const T_Position pos) const
            {
                BOOST_STATIC_ASSERT(T_Position::dim == DIM2);

                T_Position intPart;
                T_Position fracPart;

                fracPart[0] = pmacc::math::modf(pos[0], &(intPart[0]));
                fracPart[1] = pmacc::math::modf(pos[1], &(intPart[1]));

                const math::Int<DIM2> idx2D(static_cast<int>(intPart[0]), static_cast<int>(intPart[1]));

                type result = pmacc::traits::GetInitializedInstance<type>()(0.0);
                typedef typename T_Position::type PositionComp;
                for(int i = 0; i < 2; i++)
                {
                    const PositionComp weighting1D = (i == 0 ? (PositionComp(1.0) - fracPart[0]) : fracPart[0]);
                    for(int j = 0; j < 2; j++)
                    {
                        const PositionComp weighting2D
                            = weighting1D * (j == 0 ? (PositionComp(1.0) - fracPart[1]) : fracPart[1]);
                        result += static_cast<type>(weighting2D * this->cursor[idx2D + math::Int<DIM2>(i, j)]);
                    }
                }

                return result;
            }
        };

        template<typename T_Cursor>
        struct LinearInterpAccessor<T_Cursor, DIM3>
        {
            typedef T_Cursor Cursor;
            typedef typename T_Cursor::ValueType type;

            Cursor cursor;

            /**
             * @param cursor input data
             */
            HDINLINE LinearInterpAccessor(const Cursor& cursor) : cursor(cursor)
            {
            }

            template<typename T_Position>
            HDINLINE type operator()(const T_Position pos) const
            {
                BOOST_STATIC_ASSERT(T_Position::dim == DIM3);

                T_Position intPart;
                T_Position fracPart;

                fracPart[0] = pmacc::math::modf(pos[0], &(intPart[0]));
                fracPart[1] = pmacc::math::modf(pos[1], &(intPart[1]));
                fracPart[2] = pmacc::math::modf(pos[2], &(intPart[2]));

                const math::Int<DIM3> idx3D(
                    static_cast<int>(intPart[0]),
                    static_cast<int>(intPart[1]),
                    static_cast<int>(intPart[2]));

                type result = pmacc::traits::GetInitializedInstance<type>()(0.0);
                typedef typename T_Position::type PositionComp;
                for(int i = 0; i < 2; i++)
                {
                    const PositionComp weighting1D = (i == 0 ? (PositionComp(1.0) - fracPart[0]) : fracPart[0]);
                    for(int j = 0; j < 2; j++)
                    {
                        const PositionComp weighting2D
                            = weighting1D * (j == 0 ? (PositionComp(1.0) - fracPart[1]) : fracPart[1]);
                        for(int k = 0; k < 2; k++)
                        {
                            const PositionComp weighting3D
                                = weighting2D * (k == 0 ? (PositionComp(1.0) - fracPart[2]) : fracPart[2]);
                            result += static_cast<type>(weighting3D * this->cursor[idx3D + math::Int<DIM3>(i, j, k)]);
                        }
                    }
                }

                return result;
            }
        };

    } // namespace cursor
} // namespace pmacc
