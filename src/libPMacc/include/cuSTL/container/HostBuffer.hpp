/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Alexander Grund
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

#include <boost/type_traits/is_same.hpp>
#include <cuSTL/container/allocator/HostMemAllocator.hpp>
#include <cuSTL/container/copier/H2HCopier.hpp>
#include <cuSTL/container/assigner/HostMemAssigner.hpp>
#include "CartBuffer.hpp"
#include "allocator/tag.h"
#include "copier/Memcopy.hpp"

namespace PMacc
{
namespace container
{

/** typedef version of a CartBuffer for a CPU.
 * Additional feature: Able to copy data from a DeviceBuffer
 * \tparam Type type of a single datum
 * \tparam dim Dimension of the container
 */
template<typename Type, int dim>
class HostBuffer
 : public CartBuffer<Type, dim, allocator::HostMemAllocator<Type, dim>,
                                copier::H2HCopier<dim>,
                                assigner::HostMemAssigner<dim> >
{
private:
    typedef CartBuffer<Type, dim, allocator::HostMemAllocator<Type, dim>,
                                  copier::H2HCopier<dim>,
                                  assigner::HostMemAssigner<dim> > Base;
///\todo: make protected
public:
    HostBuffer() {}
public:
    /* constructors
     *
     * \param _size size of the container
     *
     * \param x,y,z convenient wrapper
     *
     */
    HostBuffer(const math::Size_t<dim>& _size) : Base(_size) {}
    HostBuffer(size_t x) : Base(x) {}
    HostBuffer(size_t x, size_t y) : Base(x, y) {}
    HostBuffer(size_t x, size_t y, size_t z) : Base(x, y, z) {}
    HostBuffer(const Base& base) : Base(base) {}

    template<typename DBuffer>
    HostBuffer& operator=(const DBuffer& rhs)
    {
        BOOST_STATIC_ASSERT((boost::is_same<typename DBuffer::memoryTag, allocator::tag::device>::value));
        BOOST_STATIC_ASSERT((boost::is_same<typename DBuffer::type, Type>::value));
        BOOST_STATIC_ASSERT(DBuffer::dim == dim);

        cudaWrapper::Memcopy<dim>()(this->dataPointer, this->pitch, rhs.getDataPointer(), rhs.getPitch(),
                                this->_size, cudaWrapper::flags::Memcopy::deviceToHost);

        return *this;
    }

    HostBuffer& operator=(const Base& rhs)
    {
        Base::operator=(rhs);
        return *this;
    }

};

} // container
} // PMacc

