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

#include "pmacc/types.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/cuSTL/cursor/compile-time/BufferCursor.hpp"
#include "pmacc/cuSTL/cursor/navigator/CartNavigator.hpp"
#include "pmacc/cuSTL/cursor/accessor/PointerAccessor.hpp"
#include "pmacc/cuSTL/zone/compile-time/SphericZone.hpp"

namespace pmacc
{
    namespace container
    {
        namespace CT
        {
            /** compile-time version of container::CartBuffer
             * \tparam _Size compile-time vector specifying the size of the container
             */
            template<typename Type, typename _Size, typename Allocator, typename Copier, typename Assigner>
            class CartBuffer
            {
            public:
                typedef Type type;
                typedef _Size Size;
                typedef typename Allocator::Pitch Pitch;
                typedef cursor::CT::BufferCursor<Type, Pitch> Cursor;
                static constexpr int dim = Size::dim;
                typedef zone::CT::SphericZone<_Size, typename math::CT::make_Int<dim, 0>::type> Zone;

            private:
                Type* dataPointer;
                // HDINLINE void init();
            public:
                template<typename T_Acc>
                DINLINE CartBuffer(T_Acc const& acc);
                DINLINE CartBuffer(const CT::CartBuffer<Type, Size, Allocator, Copier, Assigner>& other);

                DINLINE CT::CartBuffer<Type, Size, Allocator, Copier, Assigner>& operator=(
                    const CT::CartBuffer<Type, Size, Allocator, Copier, Assigner>& rhs);

                DINLINE void assign(const Type& value);
                DINLINE Type* getDataPointer() const
                {
                    return dataPointer;
                }

                DINLINE cursor::CT::BufferCursor<Type, Pitch> origin() const;
                /*
                HDINLINE Cursor<PointerAccessor<Type>, CartNavigator<dim>, char*>
                originCustomAxes(const math::UInt32<dim>& axes) const;
                */
                DINLINE math::Size_t<dim> size() const
                {
                    return math::Size_t<dim>(Size());
                }

                DINLINE Zone zone() const
                {
                    return Zone();
                }
            };

        } // namespace CT
    } // namespace container
} // namespace pmacc

#include "CartBuffer.tpp"
