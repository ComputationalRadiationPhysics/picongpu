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

#ifndef ZONE_SPHERICZONE_HPP
#define ZONE_SPHERICZONE_HPP

#include <stdint.h>
#include "math/vector/Int.hpp"
#include "math/vector/Size_t.hpp"

namespace PMacc
{
namespace zone
{

namespace tag
{
struct SphericZone {};
}

/* spheric (no holes), cartesian zone
 *
 * \tparam _dim dimension of the zone
 *
 * This is a zone which is simply described by a size and a offset.
 *
 */
template<int _dim>
struct SphericZone
{
    typedef tag::SphericZone tag;
    static const int dim = _dim;
    math::Size_t<dim> size;
    math::Int<dim> offset;

    HDINLINE SphericZone() {}
    HDINLINE SphericZone(const math::Size_t<dim>& size) : size(size), offset(math::Int<dim>(0)) {}
    HDINLINE SphericZone(const math::Size_t<dim>& size,
                         const math::Int<dim>& offset) : size(size), offset(offset) {}

    /* Returns whether pos is within the zone */
    HDINLINE bool within(const PMacc::math::Int<_dim>& pos) const
    {
        bool result = true;
        for(int i = 0; i < _dim; i++)
            if((pos[i] < offset[i]) || (pos[i] >= offset[i] + (int)size[i])) result = false;
        return result;
    }
};

} // zone
} // PMacc

#endif // ZONE_SPHERICZONE_HPP
