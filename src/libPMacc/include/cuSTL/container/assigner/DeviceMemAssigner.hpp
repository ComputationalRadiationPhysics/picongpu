/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera, Benjamin Worpitz,
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

#include "verify.hpp"
#include "cuSTL/cursor/BufferCursor.hpp"
#include "cuSTL/zone/SphericZone.hpp"
#include "cuSTL/algorithm/kernel/run-time/Foreach.hpp"
#include "lambda/Expression.hpp"
#include "math/vector/Size_t.hpp"
#include "pmacc_types.hpp"

#include <boost/math/common_factor_rt.hpp>
#include <boost/mpl/placeholders.hpp>

#include <stdint.h>

namespace PMacc
{
namespace assigner
{

namespace bmpl = boost::mpl;

template<typename T_Dim = bmpl::_1, typename T_CartBuffer = bmpl::_2>
struct DeviceMemAssigner
{
    static constexpr int dim = T_Dim::value;
    typedef T_CartBuffer CartBuffer;

    template<typename Type>
    HINLINE void assign(const Type& value)
    {
        // "Curiously recurring template pattern"
        CartBuffer* buffer = static_cast<CartBuffer*>(this);

        zone::SphericZone<dim> myZone(buffer->size());
        cursor::BufferCursor<Type, dim> cursor(buffer->dataPointer, buffer->pitch);

        /* The greatest common divisor of each component of the volume size
         * and a certain power of two value gives the best suitable block size */
        math::Size_t<3> blockDim(math::Size_t<3>::create(1));
        size_t maxValues[] = {16, 16, 4}; // maximum values for each dimension
        for(int i = 0; i < dim; i++)
        {
            blockDim[i] = boost::math::gcd(buffer->size()[i], maxValues[dim-1]);
        }
        /* the maximum number of threads per block for devices with
         * compute capability > 2.0 is 1024 */
        PMACC_VERIFY(blockDim.productOfComponents() <= 1024);

        algorithm::kernel::RT::Foreach foreach(blockDim);
        foreach(myZone, cursor, lambda::_1 = value);
    }
};

} // assigner
} // PMacc
