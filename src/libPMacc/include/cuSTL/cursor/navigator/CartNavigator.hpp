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

#ifndef CURSOR_CARTNAVIGATOR_HPP
#define CURSOR_CARTNAVIGATOR_HPP

#include <math/vector/Int.hpp>
#include "tag.h"
#include <cuSTL/cursor/traits.hpp>

namespace PMacc
{
namespace cursor
{

template<int _dim>
class CartNavigator
{
public:
    typedef tag::CartNavigator tag;
    static const int dim = _dim;
private:
    math::Int<dim> factor;
public:
    HDINLINE
    CartNavigator(math::Int<dim> factor) : factor(factor) {}

    template<typename Data>
    HDINLINE
    Data operator()(const Data& data, const math::Int<dim>& jump) const
    {
        char* result = (char*)data;
        result += dot(jump, this->factor);
        return (Data)result;
    }

    HDINLINE
    const math::Int<dim>& getFactor() const {return factor;}
};

namespace traits
{

template<int _dim>
struct dim<CartNavigator<_dim> >
{
    static const int value = _dim;
};

} // traits

} // cursor
} // PMacc

#endif // CURSOR_CARTNAVIGATOR_HPP
