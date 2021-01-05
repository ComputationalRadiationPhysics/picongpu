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
#include "pmacc/cuSTL/cursor/Cursor.hpp"
#include "pmacc/cuSTL/cursor/accessor/LinearInterpAccessor.hpp"
#include "pmacc/cuSTL/cursor/navigator/PlusNavigator.hpp"
#include "pmacc/cuSTL/cursor/traits.hpp"
#include "pmacc/math/vector/Vector.hpp"
#include "pmacc/result_of_Functor.hpp"

namespace pmacc
{
    namespace cursor
    {
        namespace tools
        {
            /** Return a cursor that does 1D, 2D or 3D, linear interpolation on input data.
             *
             * \tparam T_PositionComp integral type of the weighting factor
             */
            template<typename T_PositionComp = float>
            struct LinearInterp
            {
                template<typename T_Cursor>
                Cursor<
                    LinearInterpAccessor<T_Cursor>,
                    PlusNavigator,
                    pmacc::math::Vector<T_PositionComp, pmacc::cursor::traits::dim<T_Cursor>::value>>
                    HDINLINE operator()(const T_Cursor& cur)
                {
                    return make_Cursor(
                        LinearInterpAccessor<T_Cursor>(cur),
                        PlusNavigator(),
                        pmacc::math::Vector<T_PositionComp, pmacc::cursor::traits::dim<T_Cursor>::value>::create(0.0));
                }
            };

        } // namespace tools
    } // namespace cursor

    namespace result_of
    {
        template<typename T_Cursor, typename T_PositionComp>
        struct Functor<cursor::tools::LinearInterp<T_PositionComp>, T_Cursor>
        {
            typedef pmacc::cursor::Cursor<
                cursor::LinearInterpAccessor<T_Cursor>,
                cursor::PlusNavigator,
                pmacc::math::Vector<T_PositionComp, pmacc::cursor::traits::dim<T_Cursor>::value>>
                type;
        };

    } // namespace result_of

} // namespace pmacc
