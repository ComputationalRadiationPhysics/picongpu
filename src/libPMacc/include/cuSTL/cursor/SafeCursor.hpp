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
 
#ifndef CURSOR_SAVECURSOR_HPP
#define CURSOR_SAVECURSOR_HPP

#include <cuSTL/cursor/traits.hpp>
#include <math/vector/Int.hpp>
#include <cuSTL/cursor/Cursor.hpp>

namespace PMacc
{
namespace cursor
{

template<typename Cursor>
class SafeCursor : public Cursor
{
public:
    static const int dim = PMacc::cursor::traits::dim<Cursor>::value;
private:
    const math::Int<dim> lowerExtent;
    const math::Int<dim> upperExtent;
    math::Int<dim> offset;
    bool enabled;
public:
    HDINLINE SafeCursor(const Cursor& cursor, 
                        const math::Int<dim>& lowerExtent,
                        const math::Int<dim>& upperExtent)
        : Cursor(cursor), 
          lowerExtent(lowerExtent), 
          upperExtent(upperExtent),
          offset(math::Int<dim>(0)),
          enabled(true)
    {}
    
    HDINLINE void enableChecking() {this->enabled = true;}
    HDINLINE void disableChecking() {this->enabled = false;}
    
    HDINLINE
    typename Cursor::type operator*()
    {
        checkValidity();
        return Cursor::operator*();
    }
    
    HDINLINE
    typename boost::add_const<typename Cursor::type>::type operator*() const
    {
        checkValidity();
        return Cursor::operator*();
    }
    
    template<typename Jump>
    HDINLINE
    SafeCursor<Cursor> operator()(const Jump& jump) const
    {
        SafeCursor<Cursor> result(Cursor::operator()(jump), 
                                  this->lowerExtent,
                                  this->upperExtent);
        result.offset = this->offset + jump;
        result.enabled = this->enabled;
        return result;
    }
    
    HDINLINE
    SafeCursor<Cursor> operator()(int x) const
    {
        return (*this)(math::Int<1>(x));
    }
    
    HDINLINE
    SafeCursor<Cursor> operator()(int x, int y) const
    {
        return (*this)(math::Int<2>(x, y));
    }
    
    HDINLINE
    SafeCursor<Cursor> operator()(int x, int y, int z) const
    {
        return (*this)(math::Int<3>(x, y, z));
    }
    
    HDINLINE void operator++() {this->jump[0]++; Cursor::operator++;}
    HDINLINE void operator--() {this->jump[0]--; Cursor::operator--;}
    
    template<typename Jump>
    HDINLINE
    typename Cursor::type operator[](const Jump& jump)
    {
        return *((*this)(jump));
    }
    
    template<typename Jump>
    HDINLINE
    typename Cursor::type operator[](const Jump& jump) const
    {
        return *((*this)(jump));
    }
private:
    HDINLINE void checkValidity() const
    {
        if(!this->enabled) return;
        #pragma unroll
        for(int i = 0; i < dim; i++)
        {
            if(this->offset[i] < this->lowerExtent[i] ||
               this->offset[i] > this->upperExtent[i])
                printf("error[cursor]: index %d out of range: %d is not within [%d, %d]\n", 
                    i, this->offset[i], this->lowerExtent[i], this->upperExtent[i]);
        }
    }
};

namespace traits
{
    
template<typename Cursor>
struct dim<SafeCursor<Cursor> >
{
    static const int value = SafeCursor<Cursor>::dim;
};
    
} // traits

template<typename Cursor>
HDINLINE SafeCursor<Cursor> make_SafeCursor(
    const Cursor& cursor,
    const math::Int<traits::dim<SafeCursor<Cursor> >::value>& lowerExtent,
    const math::Int<traits::dim<SafeCursor<Cursor> >::value>& upperExtent)
{
    return SafeCursor<Cursor>(cursor, lowerExtent, upperExtent);
}

} // cursor
} // PMacc

#endif // CURSOR_SAVECURSOR_HPP
