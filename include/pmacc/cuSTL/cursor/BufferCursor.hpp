/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include "Cursor.hpp"
#include "accessor/PointerAccessor.hpp"
#include "navigator/BufferNavigator.hpp"
#include "pmacc/math/vector/Size_t.hpp"
#include "pmacc/cuSTL/cursor/traits.hpp"


namespace pmacc
{
    namespace cursor
    {
        /** The most common cursor typedef
         *
         * BufferCursor does access and jumping on a cartesian memory buffer.
         *
         * \tparam T_Type type of a single datum
         * \tparam T_dim dimension of the memory buffer
         */
        template<typename T_Type, int T_dim>
        struct BufferCursor : public Cursor<PointerAccessor<T_Type>, BufferNavigator<T_dim>, T_Type*>
        {
            /* \param pointer data pointer
             * \param pitch pitch of the memory buffer
             * pitch is a Size_t vector with one dimension less than dim
             * pitch[0] is the distance in bytes to the incremented y-coordinate
             * pitch[1] is the distance in bytes to the incremented z-coordiante
             */
            HDINLINE
            BufferCursor(T_Type* pointer, math::Size_t<T_dim - 1> pitch)
                : Cursor<PointerAccessor<T_Type>, BufferNavigator<T_dim>, T_Type*>(
                    PointerAccessor<T_Type>(),
                    BufferNavigator<T_dim>(pitch),
                    pointer)
            {
            }

            HDINLINE
            BufferCursor(const Cursor<PointerAccessor<T_Type>, BufferNavigator<T_dim>, T_Type*>& other)
                : Cursor<PointerAccessor<T_Type>, BufferNavigator<T_dim>, T_Type*>(other)
            {
            }
        };

        namespace traits
        {
            /* type trait to get the BufferCursor's dimension if it has one */
            template<typename T_Type, int T_dim>
            struct dim<BufferCursor<T_Type, T_dim>>
            {
                static constexpr int value = pmacc::cursor::traits::dim<
                    Cursor<PointerAccessor<T_Type>, BufferNavigator<T_dim>, T_Type*>>::value;
            };

        } // namespace traits

    } // namespace cursor
} // namespace pmacc
