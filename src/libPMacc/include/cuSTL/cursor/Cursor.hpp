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
 
#ifndef CURSOR_CURSOR_HPP
#define CURSOR_CURSOR_HPP

#include <boost/mpl/void.hpp>
#include "math/vector/Int.hpp"
#include "types.h"
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <cuSTL/cursor/traits.hpp>

namespace mpl = boost::mpl;

namespace PMacc
{
namespace cursor
{
    
template<typename _Accessor, typename _Navigator, typename _Marker>
class Cursor : private _Accessor, _Navigator
{
public:
    typedef typename _Accessor::type type;
    typedef typename boost::remove_reference<type>::type pureType;
    typedef _Accessor Accessor;
    typedef _Navigator Navigator;
    typedef _Marker Marker;
    typedef Cursor<Accessor, Navigator, Marker> result_type;
    typedef Cursor<Accessor, Navigator, Marker> This;
protected:
    Marker marker;
public:
    HDINLINE
    Cursor(const Accessor& accessor,
             const Navigator& navigator,
             const Marker& marker)
                : Accessor(accessor), Navigator(navigator), marker(marker) {}

    HDINLINE
    type operator*()
    {
        return Accessor::operator()(this->marker);
    }
    
    HDINLINE
    typename boost::add_const<type>::type operator*() const
    {
        return Accessor::operator()(this->marker);
    }
    
    template<typename Jump>
    HDINLINE This operator()(const Jump& jump) const
    {
        return This(getAccessor(), getNavigator(),
                    Navigator::operator()(this->marker, jump));
    }
    
    HDINLINE This operator()(int x) const
    {
        return (*this)(math::Int<1>(x));
    }
    
    HDINLINE This operator()(int x, int y) const
    {
        return (*this)(math::Int<2u>(x, y));
    }
    
    HDINLINE This operator()(int x, int y, int z) const
    {
        return (*this)(math::Int<3>(x, y, z));
    }
    
    HDINLINE void operator++() {Navigator::operator++;}
    HDINLINE void operator--() {Navigator::operator--;}
    
    template<typename Jump>
    HDINLINE
    type operator[](const Jump& jump)
    {
        return *((*this)(jump));
    }
    
    template<typename Jump>
    HDINLINE
    type operator[](const Jump& jump) const
    {
        return *((*this)(jump));
    }

    HDINLINE void enableChecking() {this->marker.enableChecking();}
    HDINLINE void disableChecking() {this->marker.disableChecking();}
    
    HDINLINE
    const _Accessor& getAccessor() const {return *this;}
    HDINLINE
    const _Navigator& getNavigator() const {return *this;}
    HDINLINE
    const Marker& getMarker() const {return this->marker;}
};

template<typename Accessor, typename Navigator, typename Marker>
HDINLINE Cursor<Accessor, Navigator, Marker> make_Cursor
(const Accessor& accessor, const Navigator& navigator, const Marker& marker)
{
    return Cursor<Accessor, Navigator, Marker>(accessor, navigator, marker);
}
           
namespace traits
{
    
template<typename _Accessor, typename _Navigator, typename _Marker>
struct dim< PMacc::cursor::Cursor<_Accessor, _Navigator, _Marker> >
{
    static const int value = PMacc::cursor::traits::dim<typename Cursor<_Accessor, _Navigator, _Marker>::Navigator >::value;
};
    
} // traits           
    
} // cursor
} // PMacc

#endif // CURSOR_CURSOR_HPP
