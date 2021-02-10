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

#include "pmacc/cuSTL/cursor/compile-time/BufferCursor.hpp"
#include "pmacc/memory/shared/Allocate.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/types.hpp"


namespace pmacc
{
    namespace allocator
    {
        namespace CT
        {
            template<typename Type, typename Size, int dim = Size::dim, int uid = 0>
            struct SharedMemAllocator;

            template<typename Type, typename Size, int uid>
            struct SharedMemAllocator<Type, Size, 1, uid>
            {
                typedef Type type;
                typedef math::CT::UInt32<> Pitch;
                static constexpr int dim = 1;
                typedef cursor::CT::BufferCursor<type, math::CT::UInt32<>> Cursor;

                template<typename T_Acc>
                DINLINE static Cursor allocate(T_Acc const& acc)
                {
                    auto& shMem = pmacc::memory::shared::
                        allocate<uid, memory::Array<Type, math::CT::volume<Size>::type::value>>(acc);
                    return Cursor(shMem.data());
                }
            };

            template<typename Type, typename Size, int uid>
            struct SharedMemAllocator<Type, Size, 2, uid>
            {
                typedef Type type;
                typedef math::CT::UInt32<sizeof(Type) * Size::x::value> Pitch;
                static constexpr int dim = 2;
                typedef cursor::CT::BufferCursor<type, Pitch> Cursor;

                template<typename T_Acc>
                DINLINE static Cursor allocate(T_Acc const& acc)
                {
                    auto& shMem = pmacc::memory::shared::
                        allocate<uid, memory::Array<Type, math::CT::volume<Size>::type::value>>(acc);
                    return Cursor(shMem.data());
                }
            };

            template<typename Type, typename Size, int uid>
            struct SharedMemAllocator<Type, Size, 3, uid>
            {
                typedef Type type;
                typedef math::CT::UInt32<sizeof(Type) * Size::x::value, sizeof(Type) * Size::x::value * Size::y::value>
                    Pitch;
                static constexpr int dim = 3;
                typedef cursor::CT::BufferCursor<type, Pitch> Cursor;

                template<typename T_Acc>
                DINLINE static Cursor allocate(T_Acc const& acc)
                {
                    auto& shMem = pmacc::memory::shared::
                        allocate<uid, memory::Array<Type, math::CT::volume<Size>::type::value>>(acc);
                    return Cursor(shMem.data());
                }
            };

        } // namespace CT
    } // namespace allocator
} // namespace pmacc
