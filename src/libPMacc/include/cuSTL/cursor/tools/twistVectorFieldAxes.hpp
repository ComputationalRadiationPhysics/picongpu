/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
#include <cuSTL/cursor/navigator/compile-time/TwistAxesNavigator.hpp>
#include <cuSTL/cursor/accessor/TwistAxesAccessor.hpp>

namespace PMacc
{
namespace cursor
{
namespace tools
{

namespace result_of
{
    
template<typename Axes, typename TCursor>
struct TwistVectorFieldAxes
{
    typedef Cursor<TwistAxesAccessor<TCursor, Axes>,
                   PMacc::cursor::CT::TwistAxesNavigator<Axes>,
                   TCursor> type;
};
    
} // result_of
  
template<typename Axes, typename TCursor>
HDINLINE
typename result_of::TwistVectorFieldAxes<Axes, TCursor>::type
twistVectorFieldAxes(const TCursor& cursor)
{
    return typename result_of::TwistVectorFieldAxes<Axes, TCursor>::type
        (TwistAxesAccessor<TCursor, Axes>(),
        PMacc::cursor::CT::TwistAxesNavigator<Axes>(),
        cursor);
}
    
} // tools
} // cursor
} // PMacc
