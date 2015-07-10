/**
 * Copyright 2013,2015 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cuSTL/cursor/Cursor.hpp>
#include <cuSTL/cursor/accessor/CursorAccessor.hpp>
#include <nvidia/reduce/Reduce.hpp>
#include <cuSTL/cursor/navigator/MapTo1DNavigator.hpp>

namespace PMacc
{
namespace algorithm
{
namespace kernel
{

/** Reduce algorithm that calls a cuda kernel
 *
 */
struct Reduce
{

    /* \param srcCursor Cursor located at the origin of the area of reduce
     * \param p_zone Zone of cells spanning the area of reduce
     * \param functor Functor with two arguments which returns the result of the reduce operation.
     *        Can also be a lambda expression.
     */
    template<typename SrcCursor, typename Zone, typename NVidiaFunctor>
    typename SrcCursor::ValueType operator()(const SrcCursor& srcCursor, const Zone& p_zone, const NVidiaFunctor& functor)
    {
        SrcCursor srcCursor_shifted = srcCursor(p_zone.offset);
        
        cursor::MapTo1DNavigator<Zone::dim> myNavi(p_zone.size);
        
        BOOST_AUTO(_srcCursor, cursor::make_Cursor(cursor::CursorAccessor<SrcCursor>(),
                                                   myNavi,
                                                   srcCursor_shifted));
        
        PMacc::nvidia::reduce::Reduce reduce(1024);
        return reduce(functor, _srcCursor, p_zone.size.productOfComponents());
    }

};

} // kernel
} // algorithm
} // PMacc
