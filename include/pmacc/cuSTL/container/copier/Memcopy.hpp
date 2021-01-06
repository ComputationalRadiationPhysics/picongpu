/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "pmacc/math/vector/Size_t.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace cuplaWrapper
    {
        namespace flags
        {
            struct Memcopy
            {
                enum Direction
                {
                    hostToDevice = 0,
                    deviceToHost,
                    hostToHost,
                    deviceToDevice
                };
            };
        } // namespace flags

        template<int dim>
        struct Memcopy;

        template<>
        struct Memcopy<1>
        {
            template<typename Type>
            void operator()(
                Type* dest,
                const math::Size_t<0>,
                const Type* source,
                const math::Size_t<0>,
                const math::Size_t<1>& size,
                flags::Memcopy::Direction direction)
            {
                const cuplaMemcpyKind kind[]
                    = {cuplaMemcpyHostToDevice,
                       cuplaMemcpyDeviceToHost,
                       cuplaMemcpyHostToHost,
                       cuplaMemcpyDeviceToDevice};
                CUDA_CHECK(cuplaMemcpy(dest, source, sizeof(Type) * size.x(), kind[direction]));
            }
        };

        template<>
        struct Memcopy<2u>
        {
            template<typename Type>
            void operator()(
                Type* dest,
                const math::Size_t<1> pitchDest,
                const Type* source,
                const math::Size_t<1> pitchSource,
                const math::Size_t<2u>& size,
                flags::Memcopy::Direction direction)
            {
                const cuplaMemcpyKind kind[]
                    = {cuplaMemcpyHostToDevice,
                       cuplaMemcpyDeviceToHost,
                       cuplaMemcpyHostToHost,
                       cuplaMemcpyDeviceToDevice};

                CUDA_CHECK(cuplaMemcpy2D(
                    dest,
                    pitchDest.x(),
                    source,
                    pitchSource.x(),
                    sizeof(Type) * size.x(),
                    size.y(),
                    kind[direction]));
            }
        };

        template<>
        struct Memcopy<3>
        {
            template<typename Type>
            void operator()(
                Type* dest,
                const math::Size_t<2u> pitchDest,
                Type* source,
                const math::Size_t<2u> pitchSource,
                const math::Size_t<3>& size,
                flags::Memcopy::Direction direction)
            {
                const cuplaMemcpyKind kind[]
                    = {cuplaMemcpyHostToDevice,
                       cuplaMemcpyDeviceToHost,
                       cuplaMemcpyHostToHost,
                       cuplaMemcpyDeviceToDevice};

                cuplaPitchedPtr pitchedPtrDest;
                pitchedPtrDest.pitch = pitchDest.x();
                pitchedPtrDest.ptr = dest;
                pitchedPtrDest.xsize = size.x() * sizeof(Type);
                pitchedPtrDest.ysize = size.y();
                cuplaPitchedPtr pitchedPtrSource;
                pitchedPtrSource.pitch = pitchSource.x();
                pitchedPtrSource.ptr = source;
                pitchedPtrSource.xsize = size.x() * sizeof(Type);
                pitchedPtrSource.ysize = size.y();

                cuplaMemcpy3DParms params;
                params.srcArray = nullptr;
                params.srcPos = make_cuplaPos(0, 0, 0);
                params.srcPtr = pitchedPtrSource;
                params.dstArray = nullptr;
                params.dstPos = make_cuplaPos(0, 0, 0);
                params.dstPtr = pitchedPtrDest;
                params.extent = make_cuplaExtent(size.x() * sizeof(Type), size.y(), size.z());
                params.kind = kind[direction];
                CUDA_CHECK(cuplaMemcpy3D(&params));
            }
        };

    } // namespace cuplaWrapper
} // namespace pmacc
