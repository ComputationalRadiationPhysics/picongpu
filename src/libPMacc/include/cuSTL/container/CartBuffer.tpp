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

#include "cuSTL/container/allocator/tag.h"
#include <iostream>
#include <eventSystem/EventSystem.hpp>

namespace PMacc
{
namespace container
{

namespace detail
{
    template<int dim> struct PitchHelper;
    template<>
    struct PitchHelper<1>
    {
        template<typename TCursor>
        HDINLINE math::Size_t<0u> operator()(const TCursor&) {return math::Size_t<0u>();}
    };
    template<>
    struct PitchHelper<2>
    {
        template<typename TCursor>
        HDINLINE math::Size_t<1> operator()(const TCursor& cursor)
        {
            return math::Size_t<1>((char*)cursor(0, 1).getMarker() - (char*)cursor.getMarker());
        }
    };
    template<>
    struct PitchHelper<3>
    {
        template<typename TCursor>
        HDINLINE math::Size_t<2> operator()(const TCursor& cursor)
        {
            return math::Size_t<2>((char*)cursor(0, 1, 0).getMarker() - (char*)cursor.getMarker(),
                                     (char*)cursor(0, 0, 1).getMarker() - (char*)cursor.getMarker());
        }
    };

    template<typename MemoryTag>
    HDINLINE void notifyEventSystem() {}

    template<>
    HDINLINE void notifyEventSystem<allocator::tag::device>()
    {
#ifndef __CUDA_ARCH__
        using namespace PMacc;
        __startOperation(ITask::TASK_CUDA);
#endif
    }

    template<>
    HDINLINE void notifyEventSystem<allocator::tag::host>()
    {
#ifndef __CUDA_ARCH__
        using namespace PMacc;
        __startOperation(ITask::TASK_HOST);
#endif
    }
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::CartBuffer
(const math::Size_t<_dim>& _size)
{
    this->_size = _size;
    init();
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::CartBuffer
(size_t x)
{
    this->_size = math::Size_t<1>(x); init();
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::CartBuffer
(size_t x, size_t y)
{
    this->_size = math::Size_t<2>(x, y); init();
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::CartBuffer
(size_t x, size_t y, size_t z)
{
    this->_size = math::Size_t<3>(x, y, z); init();
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::CartBuffer
(const CartBuffer<Type, dim, Allocator, Copier, Assigner>& other)
{
    this->dataPointer = other.dataPointer;
    this->refCount = other.refCount;
    (*this->refCount)++;
    this->_size = other._size;
    this->pitch = other.pitch;
}

#define COMMA ,

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::CartBuffer
(BOOST_RV_REF(CartBuffer<Type COMMA dim COMMA Allocator COMMA Copier COMMA Assigner>) other)
{
    this->dataPointer = 0;
    this->refCount = 0;
    *this = other;
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
void CartBuffer<Type, _dim, Allocator, Copier, Assigner>::init()
{
    typename Allocator::Cursor cursor = Allocator::allocate(this->_size);
    this->dataPointer = cursor.getMarker();
#ifndef __CUDA_ARCH__
    this->refCount = new int;
#endif
    *this->refCount = 1;
    this->pitch = detail::PitchHelper<_dim>()(cursor);
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::~CartBuffer()
{
    exit();
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
void CartBuffer<Type, _dim, Allocator, Copier, Assigner>::exit()
{
    if(!this->refCount) return;
    (*(this->refCount))--;
    if(*(this->refCount) > 0)
        return;
    Allocator::deallocate(origin());
    this->dataPointer = 0;
#ifndef __CUDA_ARCH__
    delete this->refCount;
    this->refCount = 0;
#endif
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
CartBuffer<Type, _dim, Allocator, Copier, Assigner>&
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::operator=
(const CartBuffer<Type, _dim, Allocator, Copier, Assigner>& rhs)
{
    if(this->dataPointer == rhs.dataPointer) return *this;
    Copier::copy(this->dataPointer, this->pitch, rhs.dataPointer, rhs.pitch, rhs._size);
    return *this;
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
CartBuffer<Type, _dim, Allocator, Copier, Assigner>&
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::operator=
(BOOST_RV_REF(CartBuffer<Type COMMA _dim COMMA Allocator COMMA Copier COMMA Assigner>) rhs)
{
    if(this->dataPointer == rhs.dataPointer) return *this;

    exit();
    this->dataPointer = rhs.dataPointer;
    this->refCount = rhs.refCount;
    (*this->refCount)++;
    this->_size = rhs._size;
    this->pitch = rhs.pitch;
    return *this;
}

#undef COMMA

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
View<CartBuffer<Type, _dim, Allocator, Copier, Assigner> >
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::view
(math::Int<_dim> a, math::Int<_dim> b) const
{
    a = (a + (math::Int<_dim>)this->size()) % (math::Int<_dim>)this->size();
    b = (b + (math::Int<_dim>)this->size())
            % ((math::Int<_dim>)this->size() + math::Int<_dim>(1));

    View<CartBuffer<Type, _dim, Allocator, Copier, Assigner> > result;

    result.dataPointer = &(*origin()(a));
    result._size = (math::Size_t<_dim>)(b - a);
    result.pitch = this->pitch;
    result.refCount = this->refCount;
    return result;
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
void CartBuffer<Type, _dim, Allocator, Copier, Assigner>::assign(const Type& value)
{
    Assigner::assign(this->dataPointer, this->pitch, value, this->_size);
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
cursor::BufferCursor<Type, _dim> CartBuffer<Type, _dim, Allocator, Copier, Assigner>::origin() const
{
    detail::notifyEventSystem<typename Allocator::tag>();
    return cursor::BufferCursor<Type, _dim>(this->dataPointer, this->pitch);
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
cursor::SafeCursor<cursor::BufferCursor<Type, _dim> >
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::originSafe() const
{
    return cursor::make_SafeCursor(this->origin(),
                                   math::Int<_dim>(0),
                                   math::Int<_dim>(size()));
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
cursor::Cursor<cursor::PointerAccessor<Type>, cursor::CartNavigator<_dim>, char*>
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::originCustomAxes(const math::UInt<_dim>& axes) const
{
    math::Size_t<dim> factor;
    factor[0] = sizeof(Type);
    if(dim > 1) factor[1] = this->pitch[0];
    if(dim > 2) factor[2] = this->pitch[1];
    //\todo: is the conversation from size_t to uint32_t allowed?
    math::Int<dim> customFactor;
    for(int i = 0; i < dim; i++)
        customFactor[i] = (int)factor[axes[i]];
    cursor::CartNavigator<dim> navi(customFactor);

    detail::notifyEventSystem<typename Allocator::tag>();

    return cursor::Cursor<cursor::PointerAccessor<Type>, cursor::CartNavigator<dim>, char*>
            (cursor::PointerAccessor<Type>(), navi, (char*)this->dataPointer);
}

template<typename Type, int _dim, typename Allocator, typename Copier, typename Assigner>
zone::SphericZone<_dim>
CartBuffer<Type, _dim, Allocator, Copier, Assigner>::zone() const
{
    zone::SphericZone<_dim> myZone;
    myZone.offset = math::Int<_dim>(0);
    myZone.size = this->_size;
    return myZone;
}

template<typename Type, typename Allocator, typename Copier, typename Assigner>
std::ostream& operator<<(std::ostream& s, const CartBuffer<Type, 1, Allocator, Copier, Assigner>& con)
{
    for(size_t x = 0; x < con.size().x(); x++)
        s << con.origin()[x] << " ";
    return s << std::endl;
}

template<typename Type, typename Allocator, typename Copier, typename Assigner>
std::ostream& operator<<(std::ostream& s, const CartBuffer<Type, 2, Allocator, Copier, Assigner>& con)
{
    for(size_t y = 0; y < con.size().y(); y++)
    {
        for(size_t x = 0; x < con.size().x(); x++)
            s << *con.origin()(x,y) << " ";
        s << std::endl;
    }
    return s << std::endl;
}

template<typename Type, typename Allocator, typename Copier, typename Assigner>
std::ostream& operator<<(std::ostream& s, const CartBuffer<Type, 3, Allocator, Copier, Assigner>& con)
{
    for(size_t z = 0; z < con.size().z(); z++)
    {
        for(size_t y = 0; y < con.size().y(); y++)
        {
            for(size_t x = 0; x < con.size().x(); x++)
                s << *con.origin()(x,y,z) << " ";
            s << std::endl;
        }
        s << std::endl;
    }
    return s << std::endl;
}

} // container
} // PMacc
