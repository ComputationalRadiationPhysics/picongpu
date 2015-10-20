/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cuSTL/container/allocator/DeviceMemAllocator.hpp>
#include <cuSTL/container/copier/D2DCopier.hpp>
#include <cuSTL/container/assigner/DeviceMemAssigner.hpp>
#include "cuSTL/container/CartBuffer.hpp"
#include "allocator/tag.h"

#include <memory/buffers/DeviceBuffer.hpp>
#include <memory/buffers/HostBuffer.hpp>

namespace PMacc
{
namespace container
{

/** typedef version of a CartBuffer for a GPU.
 * Additional feature: Able to copy data from a HostBuffer
 * \tparam Type type of a single datum
 * \tparam dim Dimension of the container
 */
template<typename Type, int dim>
class DeviceBuffer
 : public CartBuffer<Type, dim, allocator::DeviceMemAllocator<Type, dim>,
                                copier::D2DCopier<dim>,
                                assigner::DeviceMemAssigner<dim> >
{
private:
    typedef CartBuffer<Type, dim, allocator::DeviceMemAllocator<Type, dim>,
                                  copier::D2DCopier<dim>,
                                  assigner::DeviceMemAssigner<dim> > Base;
    typedef DeviceBuffer<Type, dim> This;

///\todo: make protected
public:
    HDINLINE DeviceBuffer() {}

    BOOST_COPYABLE_AND_MOVABLE(This)
public:
    /* constructors
     *
     * \param _size size of the container
     *
     * \param x,y,z convenient wrapper
     *
     */
    HDINLINE DeviceBuffer(const math::Size_t<dim>& _size) : Base(_size) {}
    HDINLINE DeviceBuffer(size_t x) : Base(x) {}
    HDINLINE DeviceBuffer(size_t x, size_t y) : Base(x, y) {}
    HDINLINE DeviceBuffer(size_t x, size_t y, size_t z) : Base(x, y, z) {}
    HDINLINE DeviceBuffer(const Base& base) : Base(base) {}

    template<typename HBuffer>
    HDINLINE
    DeviceBuffer& operator=(const HBuffer& rhs)
    {
        BOOST_STATIC_ASSERT((boost::is_same<typename HBuffer::memoryTag, allocator::tag::host>::value));
        BOOST_STATIC_ASSERT((boost::is_same<typename HBuffer::type, Type>::value));
        BOOST_STATIC_ASSERT(dim == HBuffer::dim);

        cudaWrapper::Memcopy<dim>()(this->dataPointer, this->pitch, rhs.getDataPointer(), rhs.getPitch(),
                                this->_size, cudaWrapper::flags::Memcopy::hostToDevice);

        return *this;
    }

    HDINLINE DeviceBuffer& operator=(const Base& rhs)
    {
        Base::operator=(rhs);
        return *this;
    }

    //friend class ::PMacc::DeviceBuffer<Type, dim>;
    //friend class ::PMacc::HostBuffer<Type, dim>;
};

} // container
} // PMacc
