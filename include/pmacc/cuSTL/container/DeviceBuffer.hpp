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

#include "pmacc/cuSTL/container/allocator/DeviceMemAllocator.hpp"
#include "pmacc/cuSTL/container/copier/D2DCopier.hpp"
#include "pmacc/cuSTL/container/assigner/DeviceMemAssigner.hpp"
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
        /** typedef version of a CartBuffer for a GPU.
         * Additional feature: Able to copy data from a HostBuffer
         * \tparam Type type of a single datum
         * \tparam T_dim Dimension of the container
         */
        template<typename Type, int T_dim>
        class DeviceBuffer
            : public CartBuffer<
                  Type,
                  T_dim,
                  allocator::DeviceMemAllocator<Type, T_dim>,
                  copier::D2DCopier<T_dim>,
                  assigner::DeviceMemAssigner<>>
        {
        private:
            typedef CartBuffer<
                Type,
                T_dim,
                allocator::DeviceMemAllocator<Type, T_dim>,
                copier::D2DCopier<T_dim>,
                assigner::DeviceMemAssigner<>>
                Base;

        protected:
            HDINLINE DeviceBuffer()
            {
            }

        public:
            typedef typename Base::PitchType PitchType;

            /* constructors
             *
             * \param _size size of the container
             *
             * \param x,y,z convenient wrapper
             *
             */
            HDINLINE DeviceBuffer(const math::Size_t<T_dim>& size) : Base(size)
            {
            }
            HDINLINE DeviceBuffer(size_t x) : Base(x)
            {
            }
            HDINLINE DeviceBuffer(size_t x, size_t y) : Base(x, y)
            {
            }
            HDINLINE DeviceBuffer(size_t x, size_t y, size_t z) : Base(x, y, z)
            {
            }
            /**
             * Creates a device buffer from a pointer with a size. Assumes dense layout (no padding)
             *
             * @param ptr Pointer to the first element
             * @param size Size of the buffer
             * @param ownMemory Set to false if the memory is only a reference and managed outside of this class
             *                  Ignored for device side creation!y
             * @param pitch Pitch in bytes (number of bytes in the lower dimensions)
             */
            HDINLINE DeviceBuffer(
                Type* ptr,
                const math::Size_t<T_dim>& size,
                bool ownMemory,
                PitchType pitch = PitchType::create(0))
            {
                this->dataPointer = ptr;
                this->_size = size;
                if(T_dim >= 2)
                    this->pitch[0] = (pitch[0]) ? pitch[0] : size.x() * sizeof(Type);
                if(T_dim == 3)
                    this->pitch[1] = (pitch[1]) ? pitch[1] : this->pitch[0] * size.y();
#ifndef __CUDA_ARCH__
                this->refCount = new int;
                *this->refCount = (ownMemory) ? 1 : 2;
#endif
            }
            HDINLINE DeviceBuffer(const Base& base) : Base(base)
            {
            }
            HDINLINE DeviceBuffer(DeviceBuffer&& obj) : Base(std::move(static_cast<Base&>(obj)))
            {
            }

            HDINLINE DeviceBuffer& operator=(DeviceBuffer&& rhs)
            {
                Base::operator=(std::move(static_cast<Base&>(rhs)));
                return *this;
            }

            template<typename HBuffer>
            HINLINE typename boost::
                enable_if<boost::is_same<typename HBuffer::memoryTag, allocator::tag::host>, DeviceBuffer&>::type
                operator=(const HBuffer& rhs)
            {
                BOOST_STATIC_ASSERT((boost::is_same<typename HBuffer::type, Type>::value));
                BOOST_STATIC_ASSERT(HBuffer::dim == T_dim);
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
                    cuplaWrapper::flags::Memcopy::hostToDevice);

                return *this;
            }

            HINLINE DeviceBuffer& operator=(const Base& rhs)
            {
                Base::operator=(rhs);
                return *this;
            }

            HINLINE DeviceBuffer& operator=(const DeviceBuffer& rhs)
            {
                Base::operator=(rhs);
                return *this;
            }
        };

    } // namespace container
} // namespace pmacc
