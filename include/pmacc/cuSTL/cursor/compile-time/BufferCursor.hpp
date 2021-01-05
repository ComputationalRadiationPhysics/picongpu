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

#include "pmacc/cuSTL/cursor/Cursor.hpp"
#include "pmacc/cuSTL/cursor/accessor/PointerAccessor.hpp"
#include "pmacc/cuSTL/cursor/navigator/compile-time/BufferNavigator.hpp"
#include "pmacc/cuSTL/cursor/traits.hpp"

namespace pmacc
{
    namespace cursor
    {
        namespace CT
        {
            /** Compile-time version of cursor::BufferCursor where pitch is a compile-time vector
             */
            template<typename Type, typename Pitch>
            struct BufferCursor : public Cursor<PointerAccessor<Type>, CT::BufferNavigator<Pitch>, Type*>
            {
                HDINLINE BufferCursor(Type* pointer)
                    : Cursor<PointerAccessor<Type>, CT::BufferNavigator<Pitch>, Type*>(
                        PointerAccessor<Type>(),
                        CT::BufferNavigator<Pitch>(),
                        pointer)
                {
                }

                HDINLINE BufferCursor(const Cursor<PointerAccessor<Type>, CT::BufferNavigator<Pitch>, Type*>& cur)
                    : Cursor<PointerAccessor<Type>, CT::BufferNavigator<Pitch>, Type*>(cur)
                {
                }
            };

        } // namespace CT

        namespace traits
        {
            template<typename Type, typename Pitch>
            struct dim<CT::BufferCursor<Type, Pitch>>
            {
                const static int value = Pitch::dim + 1;
            };

        } // namespace traits

    } // namespace cursor
} // namespace pmacc
