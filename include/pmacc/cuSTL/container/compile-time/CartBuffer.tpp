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

namespace pmacc
{
    namespace container
    {
        namespace CT
        {
            template<typename Type, typename _Size, typename Allocator, typename Copier, typename Assigner>
            template<typename T_Acc>
            DINLINE CartBuffer<Type, _Size, Allocator, Copier, Assigner>::CartBuffer(T_Acc const& acc)
            {
                this->dataPointer = Allocator::allocate(acc).getMarker();
            }

            template<typename Type, typename _Size, typename Allocator, typename Copier, typename Assigner>
            DINLINE cursor::CT::BufferCursor<Type, typename Allocator::Pitch>
            CartBuffer<Type, _Size, Allocator, Copier, Assigner>::origin() const
            {
                return cursor::CT::BufferCursor<Type, Pitch>(this->dataPointer);
            }

        } // namespace CT
    } // namespace container
} // namespace pmacc
