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

#include "Memcopy.hpp"
#include "pmacc/cuSTL/cursor/BufferCursor.hpp"
#include "pmacc/cuSTL/cursor/accessor/CursorAccessor.hpp"
#include "pmacc/cuSTL/cursor/navigator/MapTo1DNavigator.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace copier
    {
        template<int T_dim>
        struct D2DCopier
        {
            static constexpr int dim = T_dim;

            PMACC_NO_NVCC_HDWARNING /* Handled via CUDA_ARCH */
                template<typename Type>
                HDINLINE static void copy(
                    Type* dest,
                    const math::Size_t<dim - 1>& pitchDest,
                    Type* source,
                    const math::Size_t<dim - 1>& pitchSource,
                    const math::Size_t<dim>& size)
            {
#ifdef __CUDA_ARCH__
                typedef cursor::BufferCursor<Type, dim> Cursor;
                Cursor bufCursorDest(dest, pitchDest);
                Cursor bufCursorSrc(source, pitchSource);
                cursor::MapTo1DNavigator<dim> myNavi(size);

                auto srcCursor = cursor::make_Cursor(cursor::CursorAccessor<Cursor>(), myNavi, bufCursorSrc);
                auto destCursor = cursor::make_Cursor(cursor::CursorAccessor<Cursor>(), myNavi, bufCursorDest);
                size_t sizeProd = size.productOfComponents();
                for(size_t i = 0; i < sizeProd; i++)
                {
                    destCursor[i] = srcCursor[i];
                }
#else
                cuplaWrapper::Memcopy<dim>()(
                    dest,
                    pitchDest,
                    source,
                    pitchSource,
                    size,
                    cuplaWrapper::flags::Memcopy::deviceToDevice);
#endif
            }
        };

    } // namespace copier
} // namespace pmacc
