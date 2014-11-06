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

#ifndef ZONE_STAGGEREDZONE_HPP
#define ZONE_STAGGEREDZONE_HPP

#include <stdint.h>
#include "../vector/Size_t.hpp"
#include "SphericZone.hpp"

namespace PMacc
{
namespace zone
{
namespace tag
{
struct StaggeredZone {};
}

template<int _dim>
struct StaggeredZone : public SphericZone<_dim>
{
    typedef tag::StaggeredZone tag;
    math::UInt<dim> staggered;
    math::UInt<dim> staggeredOffset;
};

} // zone
} // PMacc

#endif // ZONE_STAGGEREDZONE_HPP
