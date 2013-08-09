/**
 * Copyright 2013 Heiko Burau, René Widera
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
 
#ifndef CONTAINER_CARTBUFFER_HPP
#define CONTAINER_CARTBUFFER_HPP

#include <stdint.h>
#include "types.h"
#include "math/vector/Size_t.hpp"
#include "math/vector/UInt.hpp"
#include "cuSTL/cursor/BufferCursor.hpp"
#include "cuSTL/cursor/navigator/CartNavigator.hpp"
#include "cuSTL/cursor/accessor/PointerAccessor.hpp"
#include "cuSTL/cursor/SafeCursor.hpp"
#include "cuSTL/zone/SphericZone.hpp"
#include "allocator/EmptyAllocator.hpp"
#include <boost/mpl/void.hpp>
#include <boost/move/move.hpp>
#include "cuSTL/container/view/View.hpp"

namespace PMacc
{
namespace container
{
    
template<typename Type, int _dim, typename Allocator = allocator::EmptyAllocator, 
                                  typename Copier = mpl::void_, 
                                  typename Assigner = mpl::void_>
class CartBuffer
{
public:
    typedef Type type;
    typedef CartBuffer<Type, _dim, Allocator, Copier, Assigner> This;
    static const int dim = _dim;
    typedef cursor::BufferCursor<Type, dim> Cursor;
    typedef typename Allocator::tag memoryTag;
public:
    Type* dataPointer;
    int* refCount;
    math::Size_t<dim> _size;
    math::Size_t<dim-1> pitch;
    HDINLINE void init();
    HDINLINE void exit();
    HDINLINE CartBuffer() {}
private:
    BOOST_COPYABLE_AND_MOVABLE(This)
public:
    HDINLINE CartBuffer(const math::Size_t<dim>& size);
    HDINLINE CartBuffer(size_t x);
    HDINLINE CartBuffer(size_t x, size_t y);
    HDINLINE CartBuffer(size_t x, size_t y, size_t z);
    HDINLINE CartBuffer(const This& other);
    HDINLINE CartBuffer(BOOST_RV_REF(This) other);
    HDINLINE ~CartBuffer();
    
    HDINLINE This& 
    operator=(const This& rhs);
    HDINLINE This& 
    operator=(BOOST_RV_REF(This) rhs);
    
    HDINLINE View<This>
        view(math::Int<dim> a = math::Int<dim>(0),
             math::Int<dim> b = math::Int<dim>(0)) const;
    
    HDINLINE void assign(const Type& value);
    HDINLINE Type* getDataPointer() const {return dataPointer;}
    
    HDINLINE cursor::BufferCursor<Type, dim> origin() const;
    HDINLINE cursor::SafeCursor<cursor::BufferCursor<Type, dim> > originSafe() const;
    HDINLINE cursor::Cursor<cursor::PointerAccessor<Type>, cursor::CartNavigator<dim>, char*>
    originCustomAxes(const math::UInt<dim>& axes) const;
    
    HDINLINE math::Size_t<dim> size() const {return this->_size;}
    HDINLINE math::Size_t<dim-1> getPitch() const {return this->pitch;}
    HDINLINE zone::SphericZone<dim> zone() const;
};
 
} // container
} // PMacc

#include "CartBuffer.tpp"

#endif // CONTAINER_CARTBUFFER_HPP
