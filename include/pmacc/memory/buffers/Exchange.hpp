/* Copyright 2013-2022 Rene Widera, Benjamin Worpitz
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

#include "pmacc/assert.hpp"
#include "pmacc/dimensions/GridLayout.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/eventSystem/tasks/TaskReceive.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/types.hpp"

#include <memory>


namespace pmacc
{
    /**
     * DIM-dimensional buffer used for data exchange.
     *
     * Exchange defines an interface for exchanging data between hosts.
     * Equally sized buffers are created on the device as well as on the host.
     * Exchange buffers may use parts of existing GridBuffer or
     * be used as dedicated memory.
     * @attention There will be no host double buffer available if MPI direct for PMacc is enabled.
     *
     * @tparam TYPE the datatype for internal buffers
     * @tparam DIM the dimension of the internal buffers
     */
    template<class TYPE, unsigned DIM>
    class Exchange
    {
    public:
        Exchange(
            DeviceBuffer<TYPE, DIM>& source,
            GridLayout<DIM> memoryLayout,
            DataSpace<DIM> guardingCells,
            uint32_t extype,
            uint32_t commTag,
            uint32_t area = BORDER,
            bool sizeOnDevice = false)
            : hostBuffer(nullptr)
            , deviceDoubleBuffer(nullptr)
            , exchange(extype)
            , communicationTag(commTag)
        {
            PMACC_ASSERT(!guardingCells.isOneDimensionGreaterThan(memoryLayout.getGuard()));

            DataSpace<DIM> tmp_size = memoryLayout.getDataSpaceWithoutGuarding();
            /*
              DataSpace<DIM> tmp_size = memoryLayout.getDataSpace() - memoryLayout.getGuard() -
                      memoryLayout.getGuard(); delete on each side 2xguard*/

            DataSpace<DIM> exchangeDimensions = exchangeTypeToDim(exchange);

            for(uint32_t dim = 0; dim < DIM; dim++)
            {
                if(DIM > dim && exchangeDimensions[dim] == 1)
                    tmp_size[dim] = guardingCells[dim];
            }

            /*This is only a pointer to other device data
             */
            using DeviceBuffer = DeviceBuffer<TYPE, DIM>;
            deviceBuffer = std::make_unique<DeviceBuffer>(
                source,
                tmp_size,
                exchangeTypeToOffset(exchange, memoryLayout, guardingCells, area),
                sizeOnDevice);
            if constexpr(DIM > DIM1)
            {
                /*create double buffer on gpu for faster memory transfers*/
                deviceDoubleBuffer = std::make_unique<DeviceBuffer>(tmp_size, false, true);
            }

            if(!Environment<>::get().isMpiDirectEnabled())
            {
                using HostBuffer = HostBuffer<TYPE, DIM>;
                hostBuffer = std::make_unique<HostBuffer>(tmp_size);
            }
        }

        Exchange(DataSpace<DIM> exchangeDataSpace, uint32_t extype, uint32_t commTag, bool sizeOnDevice = false)
            : hostBuffer(nullptr)
            , deviceDoubleBuffer(nullptr)
            , exchange(extype)
            , communicationTag(commTag)
        {
            using DeviceBuffer = DeviceBuffer<TYPE, DIM>;
            deviceBuffer = std::make_unique<DeviceBuffer>(exchangeDataSpace, sizeOnDevice);

            if constexpr(DIM > DIM1)
            {
                /*create double buffer on gpu for faster memory transfers*/
                deviceDoubleBuffer = std::make_unique<DeviceBuffer>(exchangeDataSpace, false, true);
            }

            if(!Environment<>::get().isMpiDirectEnabled())
            {
                using HostBuffer = HostBuffer<TYPE, DIM>;
                hostBuffer = std::make_unique<HostBuffer>(exchangeDataSpace);
            }
        }

        /**
         * specifies in returned DataSpace which dimensions exchange data
         * @param exchange the exchange mask
         * @return DIM1 DataSpace of size 3 where 1 means exchange, 0 means no exchange
         */
        DataSpace<DIM> exchangeTypeToDim(uint32_t exchange) const
        {
            DataSpace<DIM> result;

            Mask exchangeMask(exchange);

            if(exchangeMask.containsExchangeType(LEFT) || exchangeMask.containsExchangeType(RIGHT))
                result[0] = 1;

            if constexpr(DIM > DIM1)
                if(exchangeMask.containsExchangeType(TOP) || exchangeMask.containsExchangeType(BOTTOM))
                    result[1] = 1;

            if constexpr(DIM > DIM2)
                if(exchangeMask.containsExchangeType(FRONT) || exchangeMask.containsExchangeType(BACK))
                    result[2] = 1;

            return result;
        }

        virtual ~Exchange() = default;

        DataSpace<DIM> exchangeTypeToOffset(
            uint32_t exchange,
            GridLayout<DIM>& memoryLayout,
            DataSpace<DIM> guardingCells,
            uint32_t area) const
        {
            DataSpace<DIM> size = memoryLayout.getDataSpace();
            DataSpace<DIM> border = memoryLayout.getGuard();
            Mask mask(exchange);
            DataSpace<DIM> tmp_offset;
            if constexpr(DIM >= DIM1)
            {
                if(mask.containsExchangeType(RIGHT))
                {
                    tmp_offset[0] = size[0] - border[0] - guardingCells[0];
                    if(area == GUARD)
                    {
                        tmp_offset[0] += guardingCells[0];
                    }
                }
                else
                {
                    tmp_offset[0] = border[0];
                    if(area == GUARD && mask.containsExchangeType(LEFT))
                    {
                        tmp_offset[0] -= guardingCells[0];
                    }
                }
            }
            if constexpr(DIM >= DIM2)
            {
                if(mask.containsExchangeType(BOTTOM))
                {
                    tmp_offset[1] = size[1] - border[1] - guardingCells[1];
                    if(area == GUARD)
                    {
                        tmp_offset[1] += guardingCells[1];
                    }
                }
                else
                {
                    tmp_offset[1] = border[1];
                    if(area == GUARD && mask.containsExchangeType(TOP))
                    {
                        tmp_offset[1] -= guardingCells[1];
                    }
                }
            }
            if constexpr(DIM == DIM3)
            {
                if(mask.containsExchangeType(BACK))
                {
                    tmp_offset[2] = size[2] - border[2] - guardingCells[2];
                    if(area == GUARD)
                    {
                        tmp_offset[2] += guardingCells[2];
                    }
                }
                else /*all other begin from front*/
                {
                    tmp_offset[2] = border[2];
                    if(area == GUARD && mask.containsExchangeType(FRONT))
                    {
                        tmp_offset[2] -= guardingCells[2];
                    }
                }
            }

            return tmp_offset;
        }

        /**
         * Returns the exchange buffer on the host.
         *
         * @return Exchange buffer on host
         */
        HostBuffer<TYPE, DIM>& getHostBuffer()
        {
            PMACC_ASSERT(hostBuffer != nullptr);
            return *hostBuffer;
        }

        /**
         * Returns the exchange buffer on the device.
         *
         * @return Exchange buffer on device
         */
        DeviceBuffer<TYPE, DIM>& getDeviceBuffer()
        {
            PMACC_ASSERT(deviceBuffer != nullptr);
            return *deviceBuffer;
        }

        bool hasDeviceDoubleBuffer()
        {
            return deviceDoubleBuffer != nullptr;
        }

        DeviceBuffer<TYPE, DIM>& getDeviceDoubleBuffer()
        {
            PMACC_ASSERT(deviceDoubleBuffer != nullptr);
            return *deviceDoubleBuffer;
        }

        EventTask startSend()
        {
            return Environment<>::get().Factory().createTaskSend(*this);
        }

        EventTask startReceive()
        {
            return Environment<>::get().Factory().createTaskReceive(*this);
        }

        /**
         * Returns the type describing exchange directions
         *
         * @return a value describing exchange directions
         */
        uint32_t getExchangeType() const
        {
            return exchange;
        }

        /**
         * Returns the value used for tagging ('naming') communicated messages
         *
         * @return the communication tag
         */
        uint32_t getCommunicationTag() const
        {
            return communicationTag;
        }

        /**
         * Return the buffer which can be used for data exchange with MPI
         *
         * The buffer can point to device or host memory.
         */
        Buffer<TYPE, DIM>* getCommunicationBuffer()
        {
            if(Environment<>::get().isMpiDirectEnabled())
            {
                if(hasDeviceDoubleBuffer())
                    return &(getDeviceDoubleBuffer());
                else
                    return &(getDeviceBuffer());
            }

            return &(getHostBuffer());
        }

    protected:
        /** host double buffer of the exchange data
         *
         * Is always a nullptr if MPI direct is used
         */
        std::unique_ptr<HostBuffer<TYPE, DIM>> hostBuffer;

        //! This buffer is a vector which is used as message buffer for faster memcopy
        std::unique_ptr<DeviceBuffer<TYPE, DIM>> deviceDoubleBuffer;
        std::unique_ptr<DeviceBuffer<TYPE, DIM>> deviceBuffer;

        uint32_t exchange = std::numeric_limits<uint32_t>::max();
        uint32_t communicationTag = std::numeric_limits<uint32_t>::max();
    };

} // namespace pmacc
