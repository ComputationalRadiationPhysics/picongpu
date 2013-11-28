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
 
#ifndef CURSOR_MULTIINDEXCURSOR
#define CURSOR_MULTIINDEXCURSOR

#include "Cursor.hpp"
#include "navigator/MultiIndexNavigator.hpp"
#include "math/vector/Int.hpp"

namespace PMacc
{
namespace cursor
{
    
template<int dim>
HDINLINE
cursor::Cursor<cursor::MarkerAccessor<math::Int<dim> >, MultiIndexNavigator<dim>,
               math::Int<dim> >
               make_MultiIndexCursor(const math::Int<dim>& idx = math::Int<dim>(0))
{
    return make_Cursor(cursor::MarkerAccessor<math::Int<dim> >(),
                       MultiIndexNavigator<dim>(),
                       idx);
}
    
} // cursor
} // PMacc

#endif // CURSOR_MULTIINDEXCURSOR
