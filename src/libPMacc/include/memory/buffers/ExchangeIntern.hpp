/**
 * Copyright 2013-2016 Rene Widera, Benjamin Worpitz
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

#include "dimensions/GridLayout.hpp"
#include "mappings/simulation/GridController.hpp"
#include "memory/buffers/Exchange.hpp"
#include "memory/dataTypes/Mask.hpp"
#include "memory/buffers/DeviceBufferIntern.hpp"
#include "memory/buffers/HostBufferIntern.hpp"

#include "eventSystem/tasks/Factory.hpp"
#include "eventSystem/tasks/TaskReceive.hpp"
#include "assert.hpp"
#include "pmacc_types.hpp"


namespace PMacc
{

    /**
     * Internal Exchange implementation.
     */
    template <class TYPE, unsigned DIM>
    class ExchangeIntern : public Exchange<TYPE, DIM>
    {
    public:

        ExchangeIntern(DeviceBuffer<TYPE, DIM>& source, GridLayout<DIM> memoryLayout, DataSpace<DIM> guardingCells, uint32_t exchange,
                       uint32_t communicationTag, uint32_t area = BORDER, bool sizeOnDevice = false) :
        Exchange<TYPE, DIM>(exchange, communicationTag), deviceDoubleBuffer(NULL)
        {

            PMACC_ASSERT(!guardingCells.isOneDimensionGreaterThan(memoryLayout.getGuard()));

            DataSpace<DIM> tmp_size = memoryLayout.getDataSpaceWithoutGuarding();
            /*
              DataSpace<DIM> tmp_size = memoryLayout.getDataSpace() - memoryLayout.getGuard() -
                      memoryLayout.getGuard(); delete on each side 2xguard*/

            DataSpace<DIM> exchangeDimensions = exchangeTypeToDim(exchange);

            for (uint32_t dim = 0; dim < DIM; dim++)
            {
                if (DIM > dim && exchangeDimensions[dim] == 1)
                    tmp_size[dim] = guardingCells[dim];
            }

            /*This is only a pointer to other device data
             */
            this->deviceBuffer = new DeviceBufferIntern<TYPE, DIM > (source, tmp_size,
                                                                     exchangeTypeToOffset(exchange, memoryLayout, guardingCells, area),
                                                                     sizeOnDevice);
            if (DIM > DIM1)
            {
                /*create double buffer on gpu for faster memory transfers*/
                this->deviceDoubleBuffer = new DeviceBufferIntern<TYPE, DIM > (tmp_size, false, true);
            }

            this->hostBuffer = new HostBufferIntern<TYPE, DIM > (tmp_size);
        }

        ExchangeIntern(DataSpace<DIM> exchangeDataSpace, uint32_t exchange,
                       uint32_t communicationTag, bool sizeOnDevice = false) :
        Exchange<TYPE, DIM>(exchange, communicationTag), deviceDoubleBuffer(NULL)
        {
            this->deviceBuffer = new DeviceBufferIntern<TYPE, DIM > (exchangeDataSpace, sizeOnDevice);
           //  this->deviceBuffer = new DeviceBufferIntern<TYPE, DIM > (exchangeDataSpace, sizeOnDevice,true);
            if (DIM > DIM1)
            {
                /*create double buffer on gpu for faster memory transfers*/
               this->deviceDoubleBuffer = new DeviceBufferIntern<TYPE, DIM > (exchangeDataSpace, false, true);
            }

            this->hostBuffer = new HostBufferIntern<TYPE, DIM > (exchangeDataSpace);
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

            if (exchangeMask.containsExchangeType(LEFT) || exchangeMask.containsExchangeType(RIGHT))
                result[0] = 1;

            if (DIM > DIM1 && (exchangeMask.containsExchangeType(TOP) || exchangeMask.containsExchangeType(BOTTOM)))
                result[1] = 1;

            if (DIM > DIM2 && (exchangeMask.containsExchangeType(FRONT) || exchangeMask.containsExchangeType(BACK)))
                result[2] = 1;

            return result;
        }

        virtual ~ExchangeIntern()
        {
            __delete(hostBuffer);
            __delete(deviceBuffer);
            __delete(deviceDoubleBuffer);
        }

        DataSpace<DIM> exchangeTypeToOffset(uint32_t exchange, GridLayout<DIM> &memoryLayout,
                                            DataSpace<DIM> guardingCells, uint32_t area) const
        {
            DataSpace<DIM> size = memoryLayout.getDataSpace();
            DataSpace<DIM> border = memoryLayout.getGuard();
            Mask mask(exchange);
            DataSpace<DIM> tmp_offset;
            if (DIM >= DIM1)
            {
                if (mask.containsExchangeType(RIGHT))
                {
                    tmp_offset[0] = size[0] - border[0] - guardingCells[0];
                    if (area == GUARD)
                    {
                        tmp_offset[0] += guardingCells[0];
                    }
                    /* std::cout<<"offset="<<tmp_offset[0]<<"border"<<border[0]<<std::endl;*/
                }
                else
                {
                    tmp_offset[0] = border[0];
                    if (area == GUARD && mask.containsExchangeType(LEFT))
                    {
                        tmp_offset[0] -= guardingCells[0];
                    }
                }
            }
            if (DIM >= DIM2)
            {
                if (mask.containsExchangeType(BOTTOM))
                {
                    tmp_offset[1] = size[1] - border[1] - guardingCells[1];
                    if (area == GUARD)
                    {
                        tmp_offset[1] += guardingCells[1];
                    }
                }
                else
                {
                    tmp_offset[1] = border[1];
                    if (area == GUARD && mask.containsExchangeType(TOP))
                    {
                        tmp_offset[1] -= guardingCells[1];
                    }
                }
            }
            if (DIM == DIM3)
            {
                if (mask.containsExchangeType(BACK))
                {
                    tmp_offset[2] = size[2] - border[2] - guardingCells[2];
                    if (area == GUARD)
                    {
                        tmp_offset[2] += guardingCells[2];
                    }
                }
                else /*all other begin from front*/
                {
                    tmp_offset[2] = border[2];
                    if (area == GUARD && mask.containsExchangeType(FRONT))
                    {
                        tmp_offset[2] -= guardingCells[2];
                    }
                }
            }

            return tmp_offset;

        }

        virtual HostBuffer<TYPE, DIM>& getHostBuffer()
        {
            return *hostBuffer;
        }

        virtual DeviceBuffer<TYPE, DIM>& getDeviceBuffer()
        {
            return *deviceBuffer;
        }

        virtual bool hasDeviceDoubleBuffer()
        {
            return deviceDoubleBuffer != NULL;
        }

        virtual DeviceBuffer<TYPE, DIM>& getDeviceDoubleBuffer()
        {
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

    protected:
        HostBufferIntern<TYPE, DIM> *hostBuffer;

        /*! This buffer is a vector which is used as message buffer for faster memcopy
         */
        DeviceBufferIntern<TYPE, DIM> *deviceDoubleBuffer;
        DeviceBufferIntern<TYPE, DIM> *deviceBuffer;

    };

}
