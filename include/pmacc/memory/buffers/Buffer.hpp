/* Copyright 2013-2023 Heiko Burau, Rene Widera, Benjamin Worpitz
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"
#include "pmacc/types.hpp"

#include <limits>

namespace pmacc
{
    /** Minimal function description of a buffer,
     *
     * @tparam T_Type data type stored in the buffer
     * @tparam T_dim dimension of the buffer (1-3)
     */
    template<class T_Type, unsigned T_dim>
    class Buffer
    {
    protected:
        using CurrentSizeBufferHost = ::alpaka::Buf<HostDevice, size_t, AlpakaDim<DIM1>, MemIdxType>;
        CurrentSizeBufferHost currentSizeBufferHost;
        MemSpace<T_dim> m_capacityND;

    private:
        Buffer(Buffer const&) = delete;
        Buffer& operator=(Buffer const&) = delete;

    public:
        using DataBoxType = DataBox<PitchedBox<T_Type, T_dim>>;

        /** constructor
         *
         * @param size extent for each dimension (in elements)
         *             if the buffer is a view to an existing buffer the size
         *             can be less than `physicalMemorySize`
         */
        Buffer(MemSpace<T_dim> size)
            : currentSizeBufferHost(alpaka::allocMappedBufIfSupported<size_t, MemIdxType>(
                manager::Device<HostDevice>::get().current(),
                manager::Device<ComputeDevice>::get().getPlatform(),
                MemSpace<DIM1>(1).toAlpakaMemVec()))
            , m_capacityND(size)
            , isMemoryContiguous(true)
        {
            Buffer::setCurrentSize(size.productOfComponents());
        }

        virtual ~Buffer()
        {
            eventSystem::startOperation(ITask::TASK_HOST);
        }

        /** get the capacity of the buffer
         *
         * @todo should be changed to MemSpace but we need to check if the values are used by MPI calls where
         * currently int vector is assumed
         */
        DataSpace<T_dim> capacityND() const
        {
            return m_capacityND;
        }

        /** give the plane C pointer to the data
         *
         * @attention Memory must be contiguous else the call will fail in debug mode.
         *
         * @return pointer to the data
         */
        virtual T_Type* data() = 0;

        /** get number of elements
         *
         * @return count of current elements per dimension
         */
        virtual size_t getCurrentSize()
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            return alpaka::getPtrNative(this->currentSizeBufferHost)[0];
        }

        /** set total number of elements
         *
         * @param newSize number of elements per dimension
         */
        virtual void setCurrentSize(size_t const newSize)
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            PMACC_ASSERT(static_cast<size_t>(newSize) <= static_cast<size_t>(capacityND().productOfComponents()));
            alpaka::getPtrNative(this->currentSizeBufferHost)[0] = newSize;
        }

        /** Total number of elements mapped to the N-dimensional size of the buffer */
        virtual MemSpace<T_dim> getCurrentDataSpace()
        {
            return getCurrentDataSpace(getCurrentSize());
        }

        /** Spread of memory per dimension which is currently used
         *
         * @return if DIM == DIM1 than return count of elements (x-direction)
         * if DIM == DIM2 than return how many lines (y-direction) of memory is used
         * if DIM == DIM3 than return how many slides (z-direction) of memory is used
         */
        virtual MemSpace<T_dim> getCurrentDataSpace(size_t size)
        {
            MemSpace<T_dim> tmp;
            auto current_size = size;

            //!\todo: current size can be changed if it is a DeviceBuffer and current size is on device
            // call first get current size (but const not allow this)

            if constexpr(T_dim == DIM1)
            {
                tmp[0] = current_size;
            }
            if constexpr(T_dim == DIM2)
            {
                if(current_size <= m_capacityND[0])
                {
                    tmp[0] = current_size;
                    tmp[1] = 1u;
                }
                else
                {
                    tmp[0] = m_capacityND[0];
                    tmp[1] = (current_size + m_capacityND[0] - 1u) / m_capacityND[0];
                }
            }
            if constexpr(T_dim == DIM3)
            {
                if(current_size <= m_capacityND[0])
                {
                    tmp[0] = current_size;
                    tmp[1] = 1u;
                    tmp[2] = 1u;
                }
                else if(current_size <= (m_capacityND[0] * m_capacityND[1]))
                {
                    tmp[0] = m_capacityND[0];
                    tmp[1] = (current_size + m_capacityND[0] - 1u) / m_capacityND[0];
                    tmp[2] = 1u;
                }
                else
                {
                    tmp[0] = m_capacityND[0];
                    tmp[1] = m_capacityND[1];
                    tmp[2] = (current_size + (m_capacityND[0] * m_capacityND[1]) - 1u)
                        / (m_capacityND[0] * m_capacityND[1]);
                }
            }

            return tmp;
        }

        /** set all data to zero and reset current size to the capacity of the container */
        virtual void reset(bool preserveData = false) = 0;

        /** set all data to the same value
         *
         * @param value value assigned to each element of the buffer
         */
        virtual void setValue(T_Type const& value) = 0;

        /** get accessor to the container elements */
        virtual DataBox<PitchedBox<T_Type, T_dim>> getDataBox() = 0;

        /** @return true if there are no paddings between rows of the data else false */
        inline bool isContiguous()
        {
            return isMemoryContiguous;
        }

        struct CPtr
        {
            T_Type* ptr;
            size_t size;

            /** @return number of valid bytes the pointer is pointing to */
            size_t sizeInBytes() const
            {
                return size * sizeof(T_Type);
            }

            /** @return reinterprets the pointer to char* */
            char* asCharPtr() const
            {
                return reinterpret_cast<char*>(ptr);
            }
        };

        /** Get a C representation of the full memory represented by the capacity.
         *
         * Memory must be contiguous else the call will fail in debug mode.
         */
        virtual CPtr getCPtrCapacity() = 0;

        /** Get a C representation of the full memory represented by the current size.
         *
         * Memory must be contiguous else the call will fail in debug mode.
         */
        virtual CPtr getCPtrCurrentSize() = 0;

    protected:
        /** true if there are no paddings between rows else false */
        bool isMemoryContiguous = true;
    };

} // namespace pmacc
