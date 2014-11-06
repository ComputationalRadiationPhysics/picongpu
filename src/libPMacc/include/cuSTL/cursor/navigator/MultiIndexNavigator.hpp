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

#ifndef CURSOR_MULTIINDEXNAVIGATOR_HPP
#define CURSOR_MULTIINDEXNAVIGATOR_HPP

#include "math/vector/Int.hpp"
#include "tag.h"
#include <cuSTL/cursor/traits.hpp>

namespace PMacc
{
namespace cursor
{

template<int _dim>
struct MultiIndexNavigator
{
    typedef tag::MultiIndexNavigator tag;
    static const int dim = _dim;

    template<typename MultiIndex>
    HDINLINE
    MultiIndex operator()(const MultiIndex& index, const math::Int<dim>& jump) const
    {
        return index + jump;
    }
};

namespace traits
{

template<int _dim>
struct dim<MultiIndexNavigator<_dim> >
{
    static const int value = _dim;
};

}

} // cursor
} // PMacc

#endif // CURSOR_MULTIINDEXNAVIGATOR_HPP
