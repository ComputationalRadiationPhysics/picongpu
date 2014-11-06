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

#ifndef ZONE_TORICZONE_HPP
#define ZONE_TORICZONE_HPP

#include <stdint.h>
#include "../vector/Size_t.hpp"

namespace PMacc
{
namespace zone
{
namespace tag
{
struct ToricZone {};
}

template<int _dim>
struct ToricZone
{
    typedef tag::ToricZone tag;
    static const int dim = _dim;
    math::Size_t<dim> offset;
    math::Size_t<dim> size;
    uint32_t thickness;
};

} // zone
} // PMacc

#endif // ZONE_TORICCZONE_HPP

