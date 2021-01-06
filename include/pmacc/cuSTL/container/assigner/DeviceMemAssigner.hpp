/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "pmacc/verify.hpp"
#include "pmacc/cuSTL/cursor/BufferCursor.hpp"
#include "pmacc/cuSTL/zone/SphericZone.hpp"
#include "pmacc/cuSTL/algorithm/kernel/run-time/Foreach.hpp"
#include "pmacc/math/vector/Size_t.hpp"
#include "pmacc/types.hpp"

#include <pmacc/cuSTL/algorithm/functor/AssignValue.hpp>

#include <boost/integer/common_factor_rt.hpp>
#include <boost/mpl/placeholders.hpp>

#include <stdint.h>

namespace pmacc
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
                math::Size_t<3> blockSize(math::Size_t<3>::create(1));
                size_t maxValues[] = {16, 16, 4}; // maximum values for each dimension
                for(int i = 0; i < dim; i++)
                {
                    blockSize[i] = boost::integer::gcd(buffer->size()[i], maxValues[dim - 1]);
                }
                /* the maximum number of threads per block for devices with
                 * compute capability > 2.0 is 1024 */
                PMACC_VERIFY(blockSize.productOfComponents() <= 1024);

                algorithm::kernel::RT::Foreach foreach(blockSize);
                foreach(myZone, cursor, pmacc::algorithm::functor::AssignValue<Type>(value))
                    ;
            }
        };

    } // namespace assigner
} // namespace pmacc
