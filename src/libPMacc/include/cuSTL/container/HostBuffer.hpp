/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera, Alexander Grund
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
#include <exception>
#include <sstream>
#include <boost/assert.hpp>

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
                                assigner::HostMemAssigner<> >
{
private:
    typedef CartBuffer<Type, dim, allocator::HostMemAllocator<Type, dim>,
                                  copier::H2HCopier<dim>,
                                  assigner::HostMemAssigner<> > Base;
    /* makes this class able to emulate a r-value reference */
    BOOST_COPYABLE_AND_MOVABLE(HostBuffer)
protected:
    HostBuffer() {}
public:
    /* constructors
     *
     * \param _size size of the container
     *
     * \param x,y,z convenient wrapper
     *
     */
    HINLINE HostBuffer(const math::Size_t<dim>& size) : Base(size) {}
    HINLINE HostBuffer(size_t x) : Base(x) {}
    HINLINE HostBuffer(size_t x, size_t y) : Base(x, y) {}
    HINLINE HostBuffer(size_t x, size_t y, size_t z) : Base(x, y, z) {}
    /**
     * Creates a host buffer from a pointer with a size. Assumes dense layout (no padding)
     *
     * @param ptr Pointer to the first element
     * @param size Size of the buffer
     * @param ownMemory Set to false if the memory is only a reference and managed outside of this class
     */
    HINLINE HostBuffer(Type* ptr, const math::Size_t<dim>& size, bool ownMemory)
    {
        this->dataPointer = ptr;
        this->_size = size;
        if(dim >= 2)
            this->pitch[0] = size.x() * sizeof(Type);
        if(dim >= 3)
            this->pitch[1] = this->pitch[0] * size.y();
        this->refCount = new int;
        *this->refCount = (ownMemory) ? 1 : 2;
    }
    HINLINE HostBuffer(const Base& base) : Base(base) {}
    HINLINE HostBuffer(BOOST_RV_REF(This) obj): Base(boost::move(static_cast<Base&>(obj))) {}

    HINLINE This&
    operator=(BOOST_RV_REF(This) rhs)
    {
        Base::operator=(boost::move(static_cast<Base&>(rhs)));
        return *this;
    }

    template<typename DBuffer>
    HostBuffer& operator=(const DBuffer& rhs)
    {
        BOOST_STATIC_ASSERT((boost::is_same<typename DBuffer::memoryTag, allocator::tag::device>::value));
        BOOST_STATIC_ASSERT((boost::is_same<typename DBuffer::type, Type>::value));
        BOOST_STATIC_ASSERT(DBuffer::dim == dim);
        if(rhs.size() != this->size())
            throw std::invalid_argument(static_cast<std::stringstream&>(
                std::stringstream() << "Assignment: Sizes of buffers do not match: "
                    << this->size() << " <-> " << rhs.size() << std::endl).str());

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

