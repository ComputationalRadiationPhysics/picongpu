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
 
#ifndef CURSOR_BUFFERCURSOR_HPP
#define CURSOR_BUFFERCURSOR_HPP

#include "Cursor.hpp"
#include "accessor/PointerAccessor.hpp"
#include "navigator/BufferNavigator.hpp"
#include "math/vector/Size_t.hpp"
#include <cuSTL/cursor/traits.hpp>

namespace PMacc
{
namespace cursor
{
    
template<typename Type, int dim>
struct BufferCursor
 : public Cursor<PointerAccessor<Type>, BufferNavigator<dim>, Type*>
{
    HDINLINE
    BufferCursor(Type* pointer, math::Size_t<dim-1> pitch) 
     : Cursor<PointerAccessor<Type>, BufferNavigator<dim>, Type*>
            (PointerAccessor<Type>(), BufferNavigator<dim>(pitch), pointer) {}
            
    HDINLINE
    BufferCursor(const Cursor<PointerAccessor<Type>, BufferNavigator<dim>, Type*>& other)
     : Cursor<PointerAccessor<Type>, BufferNavigator<dim>, Type*>(other) {}
};

namespace traits
{
    
template<typename Type, int _dim>
struct dim<BufferCursor<Type, _dim> >
{
    static const int value = PMacc::cursor::traits::dim<
        Cursor<PointerAccessor<Type>, BufferNavigator<_dim>, Type*> >::value;
};
    
} // traits

} // cursor
} // PMacc

#endif // CURSOR_BUFFERCURSOR_HPP
