/* Copyright 2013-2019 Heiko Burau, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <stdint.h>
#include "../vector/Size_t.hpp"

namespace pmacc
{
namespace zone
{
namespace tag
{
struct ToricZone {};
}

template<int T_dim>
struct ToricZone
{
    typedef tag::ToricZone tag;
    static constexpr int dim = T_dim;
    math::Size_t<dim> offset;
    math::Size_t<dim> size;
    uint32_t thickness;
};

} // zone
} // pmacc

