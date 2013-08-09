/**
 * Copyright 2013 Felix Schmitt, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
 
/* 
 * File:   HeapBuffer.hpp
 * Author: fschmitt
 *
 * Created on 24. November 2010, 12:40
 */

#ifndef HEAPBUFFER_HPP
#define	HEAPBUFFER_HPP

#include "memory/buffers/GridBuffer.hpp"
#include "particles/memory/buffers/RingBuffer.hpp"
#include "particles/memory/boxes/HeapDataBox.hpp"
#include "eventSystem/EventSystem.hpp"

namespace PMacc
{


/**
 * Represents a RingBuffer and an actual data buffer and can generate HeapDataBoxes.
 *
 * @tparam TYPE datatype used for indexing
 * @tparam VALUE datatype used for indexed data
 * @tparam BORDERVALUE datatype used for indexed border values (TYPE by default)
 */
template <class TYPE, class VALUE, class BORDERVALUE = VALUE>
class HeapBuffer : public GridBuffer<VALUE, DIM1, BORDERVALUE>
{
public:

    enum
    {
        SizeOfOneFrame = sizeof (VALUE) + sizeof (TYPE)

    };

    /**
     * Constructor
     * \see GridBuffer
     */
    HeapBuffer(DataSpace<DIM1> dataSpace) :
    GridBuffer<VALUE, DIM1, BORDERVALUE>(dataSpace)
    {
        ringBuffer = new RingBuffer<TYPE, TYPE > (dataSpace.getElementCount());
    }

    /**
     * Destructor
     */
    virtual ~HeapBuffer()
    {
        delete ringBuffer;
        ringBuffer = NULL;
    }

    /**
     * Returns a HeapDataBox combining the host data pointer and the host RingBuffer.
     *
     * @return a HeapDataBox for the host
     */
    HeapDataBox<TYPE, VALUE> getHostHeapDataBox()
    {
        return HeapDataBox<TYPE, VALUE > (this->getHostBuffer().getBasePointer(),
                                          ringBuffer->getHostRingDataBox());
    }

    /**
     * Return a HeapDataBox combining the device data pointer and the device RingBuffer.
     *
     * @return a HeapDataBox for the device
     */
    HeapDataBox<TYPE, VALUE> getDeviceHeapDataBox()
    {
        return HeapDataBox<TYPE, VALUE > (this->getDeviceBuffer().getBasePointer(),
                                          ringBuffer->getDeviceRingDataBox());
    }

    /**
     * Copies this data and the RingBuffer data from host to device.
     */
    void hostToDevice()
    {
        __startTransaction(__getTransactionEvent());
        ringBuffer->hostToDevice();
        EventTask ev1 = __endTransaction();
        __startTransaction(__getTransactionEvent());
        GridBuffer<VALUE, DIM1, BORDERVALUE>::hostToDevice();
        __setTransactionEvent(__endTransaction() + ev1);
    }

    /**
     * Copies this data and the RingBuffer data from device to host.
     */
    void deviceToHost()
    {
        __startTransaction(__getTransactionEvent());
        ringBuffer->deviceToHost();
        EventTask ev1 = __endTransaction();

        __startTransaction(__getTransactionEvent());
        GridBuffer<VALUE, DIM1, BORDERVALUE>::deviceToHost();
        EventTask ev2 = __endTransaction();

        __setTransactionEvent(ev1 + ev2);
    }

    /**
     * Clears the internal RingBuffer on both host and device.
     */
    void clear()
    {
        ringBuffer->clear();
    }

    /**
     * Initializes the internal RingBuffer.
     */
    void initialFillBuffer()
    {
        this->reset(false);
        ringBuffer->initialFillBuffer();
    }

    /**
     * \see GridBuffer.
     */
    void addExchangeBuffer(const Mask receive, const DataSpace<DIM1> &dataSpace, uint32_t communicationTag)
    {
        this->addExchangeBuffer(receive, dataSpace, communicationTag, true);
    }

private:

    void addExchange(const Mask &receive, DataSpace<DIM1> guardingCells, uint32_t communicationTag, bool sizeOnDevice)
    {
        GridBuffer<VALUE, DIM1, BORDERVALUE>::addExchange(receive, guardingCells, sizeOnDevice);
    }

    void addExchangeBuffer(const Mask &receive, const DataSpace<DIM1> &dataSpace, uint32_t communicationTag, bool sizeOnDevice)
    {
        GridBuffer<VALUE, DIM1, BORDERVALUE>::addExchangeBuffer(receive, dataSpace, communicationTag, sizeOnDevice);
    }

protected:
    RingBuffer<TYPE, TYPE> *ringBuffer;
};
}

#endif	/* HEAPBUFFER_HPP */
