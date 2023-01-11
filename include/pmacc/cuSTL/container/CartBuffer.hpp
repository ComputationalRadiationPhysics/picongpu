/* Copyright 2013-2022 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "pmacc/cuSTL/container/view/View.hpp"
#include "pmacc/cuSTL/cursor/BufferCursor.hpp"
#include "pmacc/cuSTL/cursor/accessor/PointerAccessor.hpp"
#include "pmacc/cuSTL/cursor/navigator/CartNavigator.hpp"
#include "pmacc/cuSTL/zone/SphericZone.hpp"
#include "pmacc/math/vector/Size_t.hpp"
#include "pmacc/math/vector/UInt32.hpp"
#include "pmacc/types.hpp"

#include <boost/mpl/apply.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/void.hpp>

#include <cstdint>
#include <memory>

namespace pmacc
{
    namespace container
    {
        namespace bmpl = boost::mpl;

        /** Implementation of a box-shaped (cartesian) container type.
         * Holds a reference counter so one can have several containers sharing one buffer.
         * Is designed to be an RAII class, but does not fully obey the RAII rules (see copy-ctor).
         * The way memory gets allocated, copied and assigned is
         * fully controlled by three policy classes.
         * @tparam Type type of a single value
         * @tparam T_dim dimension of the container
         * @tparam Allocator allocates and releases memory
         * @tparam Copier copies one memory buffer to another
         * @tparam Assigner assigns a value to every datum of a memory buffer
         *
         * Assigner policy has to support `apply2`: Assigner<Dim, CartBuffer>
         *
         */
        template<
            typename Type,
            int T_dim,
            typename Allocator,
            typename Copier = mpl::void_,
            typename Assigner = bmpl::vector<bmpl::_1, bmpl::_2>>
        class CartBuffer
            : public
              /* "Curiously recurring template pattern" */
              bmpl::apply<Assigner, bmpl::int_<T_dim>, CartBuffer<Type, T_dim, Allocator, Copier, Assigner>>::type
        {
        public:
            using type = Type;
            static constexpr int dim = T_dim;
            using Cursor = cursor::BufferCursor<Type, T_dim>;
            using memoryTag = typename Allocator::tag;
            using SizeType = math::Size_t<T_dim>;
            using PitchType = math::Size_t<T_dim - 1>;

        public:
            // unchanged shared data pointer
            std::shared_ptr<Type> sharedPtr;
            // pointer to the origin of the current data view
            Type* shiftedPtr = nullptr;
            SizeType _size;
            PitchType pitch;
            HINLINE void init();
            CartBuffer() = default;

        public:
            HINLINE CartBuffer(const math::Size_t<T_dim>& size);
            HINLINE CartBuffer(size_t x);
            HINLINE CartBuffer(size_t x, size_t y);
            HINLINE CartBuffer(size_t x, size_t y, size_t z);
            CartBuffer(const CartBuffer& other) = default;
            CartBuffer(CartBuffer&& other) = default;
            ~CartBuffer() = default;

            /* copy another container into this one (hard data copy) */
            HINLINE CartBuffer& operator=(const CartBuffer& rhs);
            /* use the memory from another container and increment the reference counter */
            HINLINE CartBuffer& operator=(CartBuffer&& rhs);

            /* get a view. Views represent a clipped area of the container.
             * @param a Top left corner of the view, inside the view.
             * Negative values are remapped, e.g. Int<2>(-1,-2) == Int<2>(width-1, height-2)
             * @param b Bottom right corner of the view, outside the view.
             * Values are remapped, so that Int<2>(0,0) == Int<2>(width, height)
             */
            HINLINE View<CartBuffer> view(
                math::Int<T_dim> a = math::Int<T_dim>(0),
                math::Int<T_dim> b = math::Int<T_dim>(0)) const;

            /* get a cursor at the container's origin cell */
            HINLINE cursor::BufferCursor<Type, T_dim> origin() const;

            /* get a component-twisted cursor at the container's origin cell
             * @param axes x-axis -> axes[0], y-axis -> axes[1], ...
             * */
            HINLINE cursor::Cursor<cursor::PointerAccessor<Type>, cursor::CartNavigator<T_dim>, char*>
            originCustomAxes(const math::UInt32<T_dim>& axes) const;

            /* get a zone spanning the whole container */
            HINLINE zone::SphericZone<T_dim> zone() const;

            HINLINE Type* getDataPointer() const
            {
                return shiftedPtr;
            }
            HINLINE math::Size_t<T_dim> size() const
            {
                return this->_size;
            }
            HINLINE math::Size_t<T_dim - 1> getPitch() const
            {
                return this->pitch;
            }
            /** Returns whether the buffer has no additional pitches
             * The expected pitches are: 2D: size.x, 3D: size.x/size.x*size.y
             */
            HINLINE bool isContigousMemory() const;
        };

    } // namespace container
} // namespace pmacc

#include "CartBuffer.tpp"
