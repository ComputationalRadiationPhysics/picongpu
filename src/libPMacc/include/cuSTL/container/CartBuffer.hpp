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

/** Implementation of a box-shaped (cartesian) container type.
 * Holds a reference counter so one can have several containers sharing one buffer.
 * Is designed to be an RAII class, but does not fully obey the RAII rules (see copy-ctor).
 * The way memory gets allocated, copied and assigned is
 * fully controlled by three policy classes.
 * \tparam Type type of a single value
 * \tparam _dim dimension of the container
 * \tparam Allocator allocates and releases memory
 * \tparam Copier copies one memory buffer to another
 * \tparam Assigner assigns a value to every datum of a memory buffer
 */
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
    /* makes this class able to emulate a r-value reference */
    BOOST_COPYABLE_AND_MOVABLE(This)
public:
    HDINLINE CartBuffer(const math::Size_t<dim>& size);
    HDINLINE CartBuffer(size_t x);
    HDINLINE CartBuffer(size_t x, size_t y);
    HDINLINE CartBuffer(size_t x, size_t y, size_t z);
    /* the copy constructor just increments the reference counter but does not copy memory */
    HDINLINE CartBuffer(const This& other);
    /* the move constructor has currently the same behavior as the copy constructor */
    HDINLINE CartBuffer(BOOST_RV_REF(This) other);
    HDINLINE ~CartBuffer();
    
    /* copy another container into this one (hard data copy) */
    HDINLINE This& 
    operator=(const This& rhs);
    /* use the memory from another container and increment the reference counter */
    HDINLINE This& 
    operator=(BOOST_RV_REF(This) rhs);
    
    /* get a view. Views represent a clipped area of the container.
     * \param a Top left corner of the view, inside the view. 
     * Negative values are remapped, e.g. Int<2>(-1,-2) == Int<2>(width-1, height-2)
     * \param b Bottom right corner of the view, outside the view. 
     * Values are remapped, so that Int<2>(0,0) == Int<2>(width, height)
     */
    HDINLINE View<This>
        view(math::Int<dim> a = math::Int<dim>(0),
             math::Int<dim> b = math::Int<dim>(0)) const;
    
    /* assign value to each datum */
    PMACC_NO_NVCC_HDWARNING
    HDINLINE void assign(const Type& value);

    /* get a cursor at the container's origin cell */
    HDINLINE cursor::BufferCursor<Type, dim> origin() const;
    /* get a safe cursor at the container's origin cell */
    HDINLINE cursor::SafeCursor<cursor::BufferCursor<Type, dim> > originSafe() const;
    /* get a component-twisted cursor at the container's origin cell 
     * \param axes x-axis -> axes[0], y-axis -> axes[1], ...
     * */
    HDINLINE cursor::Cursor<cursor::PointerAccessor<Type>, cursor::CartNavigator<dim>, char*>
    originCustomAxes(const math::UInt<dim>& axes) const;
    
    /* get a zone spanning the whole container */
    HDINLINE zone::SphericZone<dim> zone() const;
    
    HDINLINE Type* getDataPointer() const {return dataPointer;}
    HDINLINE math::Size_t<dim> size() const {return this->_size;}
    HDINLINE math::Size_t<dim-1> getPitch() const {return this->pitch;}
};
 
} // container
} // PMacc

#include "CartBuffer.tpp"

#endif // CONTAINER_CARTBUFFER_HPP
