/**
 * Copyright 2013 Felix Schmitt, Rene Widera
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


#ifndef RINGBUFFER_HPP
#define	RINGBUFFER_HPP

#include "particles/frame_types.hpp"
#include "memory/buffers/GridBuffer.hpp"
#include "particles/memory/boxes/RingDataBox.hpp"
#include "dimensions/DataSpace.hpp"
#include "eventSystem/EventSystem.hpp"
#include "memory/boxes/PitchedBox.hpp"

namespace PMacc
{



/**
 * Combines two GridBuffers.
 *
 * Should be used in combination with RingDataBox
 * to provide pointers for data and pointers to start- and end-addresses.
 *
 * @tparam TYPE datatype for indexing
 * @tparam VALUE datatype for indexed data
 */
template<class TYPE, class VALUE>
class RingBuffer
{
private:

    enum
    {
        PUSH, POP, ERR
    };
public:

    /**
     * Constructor.
     *
     * Initializes indizes in RingBuffer and sets maximum size of RingBuffer if fill == true
     * (in other words: the RingBuffer is already 'full' after the constructor)
     * filling is done on host and device
     *
     * @param numElements initial size of the RingBuffer in elements of type VALUE
     * @param fill sets if the RingBuffer whould be initialized in constructor
     */
    RingBuffer(DataSpace<DIM1> numElements, bool fill = true)
    {

        ringData = new GridBuffer<VALUE, DIM1 > (numElements);
        /*memory for PUSH,POP,ERR */
        ringDataSizes = new GridBuffer<TYPE, DIM1 > (DataSpace<DIM1 > (3));

        if (fill)
        {
            initialFillBuffer();
        }

    }

    /**
     * Destructor.
     */
    virtual ~RingBuffer()
    {
        __delete(ringData);
        __delete(ringDataSizes);
    }

    /**
     * Clears the RingBuffer (resets begin and end to 0) on both host and device.
     */
    void clear()
    {
        ringDataSizes->getHostBuffer().getDataBox()[POP] = 0;
        ringDataSizes->getHostBuffer().getDataBox()[PUSH] = 0;
        ringDataSizes->hostToDevice();
    }

    /**
     * Initializes the RingBuffer.
     */
    void initialFillBuffer()
    {
        /*\todo: please fix me this is not generic, add own method for index initialisation*/
        size_t size = ringData->getGridLayout().getDataSpace().productOfComponents();
        DataBox<PitchedBox<VALUE, DIM1> > dbox = ringData->getHostBuffer().getDataBox();
        for (size_t i = 0; i < size; i++)
            dbox[i] = (VALUE) i;
        ringDataSizes->getHostBuffer().getDataBox()[POP] = 0;
        ringDataSizes->getHostBuffer().getDataBox()[PUSH] = 0;
        ringDataSizes->getHostBuffer().getDataBox()[ERR] = 0;

        hostToDevice();
    }

    /**
     * Returns a RingDataBox with TYPE addresses and VALUE values.
     *
     * Pointers of the RingDataBox are device pointers
     *
     * @return a RingDataBox, represented by this RingBuffer
     */
    RingDataBox<TYPE, VALUE> getDeviceRingDataBox()
    {
        return RingDataBox<TYPE, VALUE > (ringData->getDeviceBuffer().getBasePointer(),
                                          getSize(),
                                          ringDataSizes->getDeviceBuffer().getDataBox());
    }

    /**
     * Returns a RingDataBox with TYPE addresses and VALUE values.
     *
     * Pointers of the RingDataBox are host pointers
     * @return a RingDataBox, represented by this RingBuffer
     */
    RingDataBox<TYPE, VALUE> getHostRingDataBox()
    {
        return RingDataBox<TYPE, VALUE > (ringData->getHostBuffer().getBasePointer(),
                                          getSize(),
                                          ringDataSizes->getHostBuffer().getDataBox()); // this is the second element of the GridBuffer ringDataSizes
    }

    /**
     * Copies data and additional pointers from host to device.
     */
    void hostToDevice()
    {
        __startTransaction(__getTransactionEvent());
        ringDataSizes->hostToDevice();
        EventTask ev1 = __endTransaction();
        __startTransaction(__getTransactionEvent());
        ringData->hostToDevice();
        __setTransactionEvent(__endTransaction() + ev1);
    }

    /**
     * Copies data and additional pointers from device to host.
     */
    void deviceToHost()
    {
        __startTransaction(__getTransactionEvent());
        ringDataSizes->deviceToHost();
        EventTask ev1 = __endTransaction();
        __startTransaction(__getTransactionEvent());
        ringData->deviceToHost();
        __setTransactionEvent(__endTransaction() + ev1);
    }

    /**
     * Returns the current size of the data's host buffer.
     *
     * @return current size of host buffer of data
     */
    size_t getSize()
    {
        return ringData->getHostBuffer().getDataSpace().productOfComponents();
    }
private:
    GridBuffer<VALUE, DIM1> *ringData;
    GridBuffer<TYPE, DIM1> *ringDataSizes;
};
}

#endif	/* RINGBUFFER_HPP */
