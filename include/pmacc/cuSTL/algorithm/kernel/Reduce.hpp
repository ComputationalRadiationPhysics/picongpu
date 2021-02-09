/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include "pmacc/cuSTL/cursor/Cursor.hpp"
#include "pmacc/cuSTL/cursor/accessor/CursorAccessor.hpp"
#include "pmacc/nvidia/reduce/Reduce.hpp"
#include "pmacc/cuSTL/cursor/navigator/MapTo1DNavigator.hpp"

namespace pmacc
{
    namespace algorithm
    {
        namespace kernel
        {
            /** Reduce algorithm that calls a cupla kernel
             *
             */
            struct Reduce
            {
                /* \param srcCursor Cursor located at the origin of the area of reduce
                 * \param p_zone Zone of cells spanning the area of reduce
                 * \param functor Functor with two arguments which returns the result of the reduce operation.
                 */
                template<typename SrcCursor, typename Zone, typename NVidiaFunctor>
                typename SrcCursor::ValueType operator()(
                    const SrcCursor& srcCursor,
                    const Zone& p_zone,
                    const NVidiaFunctor& functor)
                {
                    SrcCursor srcCursor_shifted = srcCursor(p_zone.offset);

                    cursor::MapTo1DNavigator<Zone::dim> myNavi(p_zone.size);

                    auto _srcCursor
                        = cursor::make_Cursor(cursor::CursorAccessor<SrcCursor>(), myNavi, srcCursor_shifted);

                    pmacc::nvidia::reduce::Reduce reduce(1024);
                    return reduce(functor, _srcCursor, p_zone.size.productOfComponents());
                }
            };

        } // namespace kernel
    } // namespace algorithm
} // namespace pmacc
