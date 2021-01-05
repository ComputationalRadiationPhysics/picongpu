/* Copyright 2013-2021 Heiko Burau, Rene Widera, Alexander Grund
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

#include "pmacc/cuSTL/container/allocator/HostMemAllocator.hpp"
#include "pmacc/cuSTL/container/copier/H2HCopier.hpp"
#include "pmacc/cuSTL/container/assigner/HostMemAssigner.hpp"
#include "pmacc/cuSTL/container/CartBuffer.hpp"
#include "pmacc/cuSTL/container/allocator/tag.hpp"
#include "pmacc/cuSTL/container/copier/Memcopy.hpp"

#include <boost/assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>

#include <exception>
#include <sstream>
#include <utility>


namespace pmacc
{
    namespace container
    {
        /** typedef version of a CartBuffer for a CPU.
         * Additional feature: Able to copy data from a DeviceBuffer
         * \tparam Type type of a single datum
         * \tparam T_dim Dimension of the container
         */
        template<typename Type, int T_dim>
        class HostBuffer
            : public CartBuffer<
                  Type,
                  T_dim,
                  allocator::HostMemAllocator<Type, T_dim>,
                  copier::H2HCopier<T_dim>,
                  assigner::HostMemAssigner<>>
        {
        private:
            using Base = CartBuffer<
                Type,
                T_dim,
                allocator::HostMemAllocator<Type, T_dim>,
                copier::H2HCopier<T_dim>,
                assigner::HostMemAssigner<>>;

        protected:
            HostBuffer()
            {
            }

        public:
            using PitchType = typename Base::PitchType;

            /* constructors
             *
             * \param _size size of the container
             *
             * \param x,y,z convenient wrapper
             *
             */
            HINLINE HostBuffer(const math::Size_t<T_dim>& size) : Base(size)
            {
            }
            HINLINE HostBuffer(size_t x) : Base(x)
            {
            }
            HINLINE HostBuffer(size_t x, size_t y) : Base(x, y)
            {
            }
            HINLINE HostBuffer(size_t x, size_t y, size_t z) : Base(x, y, z)
            {
            }
            /**
             * Creates a host buffer from a pointer with a size. Assumes dense layout (no padding)
             *
             * @param ptr Pointer to the first element
             * @param size Size of the buffer
             * @param ownMemory Set to false if the memory is only a reference and managed outside of this class
             * @param pitch Pitch in bytes (number of bytes in the lower dimensions)
             */
            HINLINE HostBuffer(
                Type* ptr,
                const math::Size_t<3>& size,
                bool ownMemory,
                math::Size_t<2> pitch = math::Size_t<2>::create(0))
            {
                this->dataPointer = ptr;
                this->_size = size;
                this->pitch[0] = (pitch[0]) ? pitch[0] : size.x() * sizeof(Type);
                this->pitch[1] = (pitch[1]) ? pitch[1] : this->pitch[0] * size.y();
                this->refCount = new int;
                *this->refCount = (ownMemory) ? 1 : 2;
            }
            HINLINE HostBuffer(
                Type* ptr,
                const math::Size_t<2>& size,
                bool ownMemory,
                math::Size_t<1> pitch = math::Size_t<1>::create(0))
            {
                this->dataPointer = ptr;
                this->_size = size;
                this->pitch[0] = (pitch[0]) ? pitch[0] : size.x() * sizeof(Type);
                this->refCount = new int;
                *this->refCount = (ownMemory) ? 1 : 2;
            }
            HINLINE HostBuffer(Type* ptr, const math::Size_t<1>& size, bool ownMemory)
            {
                this->dataPointer = ptr;
                this->_size = size;
                // intentionally uninitialized and not RT accessible via []
                // this->pitch = pitch;
                this->refCount = new int;
                *this->refCount = (ownMemory) ? 1 : 2;
            }
            HINLINE HostBuffer(const Base& base) : Base(base)
            {
            }
            HINLINE HostBuffer(HostBuffer&& obj) : Base(std::move(static_cast<Base&>(obj)))
            {
            }

            HINLINE HostBuffer& operator=(HostBuffer&& rhs)
            {
                Base::operator=(std::move(static_cast<Base&>(rhs)));
                return *this;
            }

            template<typename DBuffer>
            HINLINE typename boost::
                enable_if<boost::is_same<typename DBuffer::memoryTag, allocator::tag::device>, HostBuffer&>::type
                operator=(const DBuffer& rhs)
            {
                BOOST_STATIC_ASSERT((boost::is_same<typename DBuffer::type, Type>::value));
                BOOST_STATIC_ASSERT(DBuffer::dim == T_dim);
                if(rhs.size() != this->size())
                    throw std::invalid_argument(static_cast<std::stringstream&>(
                                                    std::stringstream()
                                                    << "Assignment: Sizes of buffers do not match: " << this->size()
                                                    << " <-> " << rhs.size() << std::endl)
                                                    .str());

                cuplaWrapper::Memcopy<T_dim>()(
                    this->dataPointer,
                    this->pitch,
                    rhs.getDataPointer(),
                    rhs.getPitch(),
                    this->_size,
                    cuplaWrapper::flags::Memcopy::deviceToHost);

                return *this;
            }

            HINLINE HostBuffer& operator=(const Base& rhs)
            {
                Base::operator=(rhs);
                return *this;
            }

            HINLINE HostBuffer& operator=(const HostBuffer& rhs)
            {
                Base::operator=(rhs);
                return *this;
            }
        };

    } // namespace container
} // namespace pmacc
