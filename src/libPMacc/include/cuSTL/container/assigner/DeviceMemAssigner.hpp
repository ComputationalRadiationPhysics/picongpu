/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
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

#pragma once

#include "cuSTL/cursor/BufferCursor.hpp"
#include "cuSTL/zone/SphericZone.hpp"
#include "cuSTL/algorithm/kernel/run-time/Foreach.hpp"
#include "lambda/Expression.hpp"
#include "math/vector/Size_t.hpp"
#include "types.h"

#include <boost/math/common_factor.hpp>

#include <cassert>
#include <stdint.h>

namespace PMacc
{
namespace assigner
{

template<int T_dim>
struct DeviceMemAssigner
{
    BOOST_STATIC_CONSTEXPR int dim = T_dim;

    template<typename Type>
    HDINLINE static void assign(
        Type* data,
        const math::Size_t<dim-1>& pitch,
        const Type& value,
        const math::Size_t<dim>& size)
    {
#ifdef __CUDA_ARCH__
        /* The HostmemAssigner iterates over the entries and assigns them
         * This also works on the device (in a kernel) so we just use it here
         * instead of implementing it again */
        HostMemAssigner<dim>::assign(data, pitch, value, size);
#else
        zone::SphericZone<dim> myZone(size);
        cursor::BufferCursor<Type, dim> cursor(data, pitch);

        /* The greatest common divisor of each component of the volume size
         * and a certain power of two value gives the best suitable block size */
        boost::math::gcd_evaluator<size_t> gcd; // greatest common divisor
        math::Size_t<3> blockDim(math::Size_t<3>::create(1));
        int maxValues[] = {16, 16, 4}; // maximum values for each dimension
        for(int i = 0; i < dim; i++)
        {
            blockDim[i] = gcd(size[i], maxValues[dim-1]);
        }
        /* the maximum number of threads per block for devices with
         * compute capability > 2.0 is 1024 */
        assert(blockDim.productOfComponents() <= 1024);

        algorithm::kernel::RT::Foreach foreach(blockDim);
        foreach(myZone, cursor, lambda::_1 = value);
#endif
    }
};

} // assigner
} // PMacc
