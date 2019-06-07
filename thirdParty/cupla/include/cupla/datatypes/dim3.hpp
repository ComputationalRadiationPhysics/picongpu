/* Copyright 2015-2016 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include "cupla/namespace.hpp"
#include "cupla/types.hpp"
#include "cupla/datatypes/uint.hpp"

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{

    struct dim3
    {
        IdxType x, y, z;

        ALPAKA_FN_HOST_ACC
        dim3(
            IdxType vx = 1,
            IdxType vy = 1,
            IdxType vz = 1
        ) :
            x(vx),
            y(vy),
            z(vz)
        {}

        ALPAKA_FN_HOST_ACC
        dim3(
            const uint3& v
        ) :
            x(v.x),
            y(v.y),
            z(v.z)
        {}

        ALPAKA_FN_HOST_ACC
        operator uint3(void)
        {
          uint3 t;
          t.x = x;
          t.y = y;
          t.z = z;
          return t;
        }
    };

} //namespace CUPLA_ACCELERATOR_NAMESPACE
} //namespace cupla
