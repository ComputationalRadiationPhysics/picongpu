/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Rene Widera
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

#ifndef PARTICLESBUFFER_HPP
#define	PARTICLESBUFFER_HPP

#include "particles/frame_types.hpp"
#include "memory/buffers/GridBuffer.hpp"
#include "particles/memory/boxes/ParticlesBox.hpp"
#include "particles/memory/buffers/HeapBuffer.hpp"
#include "particles/memory/boxes/HeapDataBox.hpp"
#include "dimensions/GridLayout.hpp"
#include "memory/dataTypes/Mask.hpp"
#include "particles/memory/buffers/StackExchangeBuffer.hpp"
#include "eventSystem/EventSystem.hpp"
#include "particles/memory/dataTypes/SuperCell.hpp"

#include "dimensions/TVec.h"

#include "particles/boostExtension/InheritGenerators.hpp"
#include "compileTime/conversion/MakeSeq.hpp"


#include <boost/mpl/vector.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/back_inserter.hpp>

#include "particles/memory/frames/Frame.hpp"
#include "particles/Identifier.hpp"
#include "particles/memory/dataTypes/StaticArray.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>

namespace PMacc
{

/**
 * Describes DIM-dimensional buffer for particles data on the host.
 *
 * @tParam PositionType type of prosition 
 * @tparam UserTypeList typelist of user classes for particle operations
 * @tparam SuperCellSize TVec which descripe size of a superce
 * @tparam DIM dimension of the buffer (1-3)
 */
template<typename T_ParticleDescription, class SuperCellSize_, unsigned DIM>
class ParticlesBuffer
{
public:

    template<typename Key>
    struct OperatorCreatePairStaticArrayWithSuperCellSize
    {
        typedef
        bmpl::pair<Key,
        StaticArray<typename Key::type, SuperCellSize_::elements> >
        type;
    };

    template<typename Key>
    struct OperatorCreatePairStaticArrayOneElement
    {
        typedef
        bmpl::pair<Key,
        StaticArray<typename Key::type, 1u > >
        type;
    };


    typedef ExchangeMemoryIndex<vint_t, DIM - 1 > PopPushType;

    typedef SuperCellSize_ SuperCellSize;

    typedef
    typename MakeSeq<
    typename T_ParticleDescription::ValueTypeSeq,
    localCellIdx,
    multiMask
    >::type full_particleList;

    typedef
    typename MakeSeq<
    typename T_ParticleDescription::ValueTypeSeq,
    localCellIdx
    >::type border_particleList;

    typedef Frame<
    OperatorCreatePairStaticArrayWithSuperCellSize, full_particleList,
    typename T_ParticleDescription::MethodsList,
    typename T_ParticleDescription::FlagsList> ParticleType;

    typedef Frame<OperatorCreatePairStaticArrayOneElement,
    border_particleList,
    typename T_ParticleDescription::MethodsList,
    typename T_ParticleDescription::FlagsList> ParticleTypeBorder;


private:

    /*this is only for internel calculations*/
    enum
    {
        SizeOfOneBorderElement = (sizeof (ParticleTypeBorder) + sizeof (PopPushType)),
        /*size=HeapBuffer overhead+prevFrameIndex+nextFrameIndex*/
        SizeOfOneFrame = HeapBuffer<vint_t, ParticleType, ParticleTypeBorder >::SizeOfOneFrame + 2 * sizeof (vint_t)
    };

public:

    /**
     * Constructor.
     *
     * @param layout number of cell per dimension         
     * @param superCellSize size of one super cell
     * @param gpuMemory how many memory on device is used for this instance (in byte)
     */
    ParticlesBuffer(DataSpace<DIM> layout, DataSpace<DIM> superCellSize) :
    superCellSize(superCellSize), gridSize(layout), frames(NULL), framesExchanges(NULL), nextFrames(NULL), prevFrames(NULL)
    {

        exchangeMemoryIndexer = new GridBuffer<PopPushType, DIM1 > (DataSpace<DIM1 > (0));
        framesExchanges = new GridBuffer< ParticleType, DIM1, ParticleTypeBorder > (DataSpace<DIM1 > (0));

        //std::cout << "size: " << sizeof (ParticleType) << " " << sizeof (ParticleTypeBorder) << std::endl;
        DataSpace<DIM> superCellsCount = gridSize / superCellSize;

        superCells = new GridBuffer<SuperCell<vint_t>, DIM > (superCellsCount);

    }

    void createParticleBuffer(size_t gpuMemory)
    {

        numFrames = gpuMemory / SizeOfOneFrame;

        frames = new HeapBuffer<vint_t, ParticleType, ParticleTypeBorder > (DataSpace<DIM1 > (numFrames));

        nextFrames = new GridBuffer<vint_t, DIM1 > (DataSpace<DIM1 > (numFrames));
        prevFrames = new GridBuffer<vint_t, DIM1 > (DataSpace<DIM1 > (numFrames));

        /** \fixme use own log level, like log<ggLog::MEMORY >
         */
        std::cout << "mem for particles=" << gpuMemory / 1024 / 1024 << " MiB = " <<
            numFrames << " Frames = " <<
            numFrames * superCellSize.productOfComponents() <<
            " Particles" << std::endl;

        reset();
    }

    /**
     * Destructor.
     */
    virtual ~ParticlesBuffer()
    {
        delete superCells;
        delete frames;
        delete framesExchanges;
        delete nextFrames;
        delete prevFrames;
        delete exchangeMemoryIndexer;
    }

    /**
     * Resets all internal buffers.
     */
    void reset()
    {
        __startTransaction(__getTransactionEvent());
        frames->reset(false);
        frames->initialFillBuffer();
        EventTask ev1 = __endTransaction();
        __startTransaction(__getTransactionEvent());
        superCells->getDeviceBuffer().setValue(SuperCell<vint_t > ());
        superCells->getHostBuffer().setValue(SuperCell<vint_t > ());

        /*nextFrames->getDeviceBuffer().setValue(INV_IDX);//!\todo: is this needed? On device we set any new frame values to INVALID_INDEX
        prevFrames->getDeviceBuffer().setValue(INV_IDX);//!\todo: is this needed? On device we set any new frame values to INVALID_INDEX
        nextFrames->getHostBuffer().setValue(INV_IDX);//!\todo: is this needed? On device we set any new frame values to INVALID_INDEX
        prevFrames->getHostBuffer().setValue(INV_IDX);//!\todo: is this needed? On device we set any new frame values to INVALID_INDEX
         */
        __setTransactionEvent(__endTransaction() + ev1);
    }

    /**
     * Adds an exchange buffer to frames.
     *
     * @param receive Mask describing receive directions
     * @param usedMemory memory to be used for this exchange
     */
    void addExchange(Mask receive, size_t usedMemory, uint32_t communicationTag)
    {

        size_t numBorderFrames = usedMemory / SizeOfOneBorderElement;

        framesExchanges->addExchangeBuffer(receive, DataSpace<DIM1 > (numBorderFrames), communicationTag, true);

        exchangeMemoryIndexer->addExchangeBuffer(receive, DataSpace<DIM1 > (numBorderFrames), communicationTag | (1u << (20 - 5)), true);
    }

    /**
     * Returns a ParticlesBox for host frame data.
     *
     * @return host frames ParticlesBox
     */
    ParticlesBox<ParticleType, DIM> getHostParticleBox()
    {

        return ParticlesBox<ParticleType, DIM > (
                                                 superCells->getHostBuffer().getDataBox(),
                                                 frames->getHostHeapDataBox(),
                                                 VectorDataBox<vint_t > (nextFrames->getHostBuffer().getBasePointer()),
                                                 VectorDataBox<vint_t > (prevFrames->getHostBuffer().getBasePointer()));
    }

    /**
     * Returns a ParticlesBox for device frame data.
     *
     * @return device frames ParticlesBox
     */
    ParticlesBox<ParticleType, DIM> getDeviceParticleBox()
    {

        return ParticlesBox<ParticleType, DIM > (
                                                 superCells->getDeviceBuffer().getDataBox(),
                                                 frames->getDeviceHeapDataBox(),
                                                 VectorDataBox<vint_t > (nextFrames->getDeviceBuffer().getBasePointer()),
                                                 VectorDataBox<vint_t > (prevFrames->getDeviceBuffer().getBasePointer()));
    }

    /**
     * Returns if the buffer has a send exchange in ex direction.
     *
     * @param ex direction to query
     * @return true if buffer has send exchange for ex
     */
    bool hasSendExchange(uint32_t ex)
    {

        return framesExchanges->hasSendExchange(ex);
    }

    /**
     * Returns if the buffer has a receive exchange in ex direction.
     *
     * @param ex direction to query
     * @return true if buffer has receive exchange for ex
     */
    bool hasReceiveExchange(uint32_t ex)
    {

        return framesExchanges->hasReceiveExchange(ex);
    }

    StackExchangeBuffer<ParticleTypeBorder, PopPushType, DIM - 1 > getSendExchangeStack(uint32_t ex)
    {

        return StackExchangeBuffer<ParticleTypeBorder, PopPushType, DIM - 1 >
            (framesExchanges->getSendExchange(ex), exchangeMemoryIndexer->getSendExchange(ex));
    }

    StackExchangeBuffer<ParticleTypeBorder, PopPushType, DIM - 1 > getReceiveExchangeStack(uint32_t ex)
    {

        return StackExchangeBuffer<ParticleTypeBorder, PopPushType, DIM - 1 >
            (framesExchanges->getReceiveExchange(ex), exchangeMemoryIndexer->getReceiveExchange(ex));
    }

    /**
     * Starts sync data from own device buffer to neighbor device buffer.
     *
     * GridBuffer
     *
     */
    EventTask asyncCommunication(EventTask serialEvent)
    {

        return framesExchanges->asyncCommunication(serialEvent) +
            exchangeMemoryIndexer->asyncCommunication(serialEvent);
    }

    EventTask asyncSendParticles(EventTask serialEvent, uint32_t ex, EventTask &gpuFree)
    {
        /*store every gpu free event seperat to avoid raceconditions*/
        EventTask framesExchangesGPUEvent;
        EventTask exchangeMemoryIndexerGPUEvent;
        EventTask returnEvent = framesExchanges->asyncSend(serialEvent, ex, framesExchangesGPUEvent) +
            exchangeMemoryIndexer->asyncSend(serialEvent, ex, exchangeMemoryIndexerGPUEvent);
        gpuFree = framesExchangesGPUEvent + exchangeMemoryIndexerGPUEvent;
        return returnEvent;
    }

    EventTask asyncReceiveParticles(EventTask serialEvent, uint32_t ex)
    {

        return framesExchanges->asyncReceive(serialEvent, ex) +
            exchangeMemoryIndexer->asyncReceive(serialEvent, ex);
    }

    /**
     * Starts copying data from host to device.
     */
    void hostToDevice()
    {

        __startTransaction(__getTransactionEvent());
        frames->hostToDevice();
        EventTask ev1 = __endTransaction();

        __startTransaction(__getTransactionEvent());
        superCells->hostToDevice();
        EventTask ev2 = __endTransaction();

        __startTransaction(__getTransactionEvent());
        nextFrames->hostToDevice();
        EventTask ev3 = __endTransaction();

        __startTransaction(__getTransactionEvent());
        prevFrames->hostToDevice();
        EventTask ev4 = __endTransaction();

        __setTransactionEvent(ev1 + ev2 + ev3 + ev4);
    }

    /**
     * Starts copying data from device to host.
     */
    void deviceToHost()
    {

        __startTransaction(__getTransactionEvent());
        frames->deviceToHost();
        EventTask ev1 = __endTransaction();

        __startTransaction(__getTransactionEvent());
        superCells->deviceToHost();
        EventTask ev2 = __endTransaction();

        __startTransaction(__getTransactionEvent());
        nextFrames->deviceToHost();
        EventTask ev3 = __endTransaction();

        __startTransaction(__getTransactionEvent());
        prevFrames->deviceToHost();
        EventTask ev4 = __endTransaction();

        __setTransactionEvent(ev1 + ev2 + ev3 + ev4);
    }

    /**
     * Returns number of supercells in each dimension.
     *
     * @return number of supercells
     */
    DataSpace<DIM> getSuperCellsCount()
    {

        assert(superCells != NULL);
        return superCells->getGridLayout().getDataSpace();
    }

    /**
     * Returns number of supercells in each dimension.
     *
     * @return number of supercells
     */
    GridLayout<DIM> getSuperCellsLayout()
    {

        assert(superCells != NULL);
        return superCells->getGridLayout();
    }

    /**
     * Returns size of supercells in each dimension.
     *
     * @return size of supercells
     */
    DataSpace<DIM> getSuperCellSize()
    {

        return superCellSize;
    }

    /**
     * Returns number of frames.
     *
     * @return number of frames
     */
    size_t getFrameCount()
    {
        return numFrames;
    }

private:
    GridBuffer<PopPushType, DIM1> *exchangeMemoryIndexer;

    GridBuffer<SuperCell<vint_t>, DIM> *superCells;
    GridBuffer<vint_t, DIM1> *nextFrames;
    GridBuffer<vint_t, DIM1> *prevFrames;
    /*gridbuffer for hold borderFrames, we need a own buffer to create first exchanges without core momory*/
    GridBuffer< ParticleType, DIM1, ParticleTypeBorder> *framesExchanges;
    HeapBuffer<vint_t, ParticleType, ParticleTypeBorder> *frames;

    DataSpace<DIM> superCellSize;
    DataSpace<DIM> gridSize;

    size_t numFrames;

};
}

#endif	/* PARTICLESBUFFER_HPP */