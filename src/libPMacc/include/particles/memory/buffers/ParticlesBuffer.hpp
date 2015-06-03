/**
 * Copyright 2013-2015 Axel Huebl, Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "particles/frame_types.hpp"
#include "memory/buffers/GridBuffer.hpp"
#include "particles/memory/boxes/ParticlesBox.hpp"
#include "dimensions/GridLayout.hpp"
#include "memory/dataTypes/Mask.hpp"
#include "particles/memory/buffers/StackExchangeBuffer.hpp"
#include "eventSystem/EventSystem.hpp"
#include "particles/memory/dataTypes/SuperCell.hpp"

#include "math/Vector.hpp"

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
#include "particles/ParticleDescription.hpp"


namespace PMacc
{

/**
 * Describes DIM-dimensional buffer for particles data on the host.
 *
 * @tParam T_ParticleDescription Object which describe a frame @see ParticleDescription.hpp
 * @tparam SuperCellSize_ TVec which descripe size of a superce
 * @tparam DIM dimension of the buffer (1-3)
 */
template<typename T_ParticleDescription, class SuperCellSize_, unsigned DIM>
class ParticlesBuffer
{
public:

    /** create static array
     */
    template<uint32_t T_size>
    struct OperatorCreatePairStaticArray
    {

        template<typename X>
        struct apply
        {
            typedef
            bmpl::pair<X,
            StaticArray< typename traits::Resolve<X>::type::type, bmpl::integral_c<uint32_t, T_size> >
            > type;
        };
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

    typedef
    typename ReplaceValueTypeSeq<T_ParticleDescription, full_particleList>::type
    ParticleDescriptionDefault;

    typedef Frame<
    OperatorCreatePairStaticArray<PMacc::math::CT::volume<SuperCellSize>::type::value >, ParticleDescriptionDefault> ParticleType;

    typedef
    typename ReplaceValueTypeSeq<T_ParticleDescription, border_particleList>::type
    ParticleDescriptionBorder;

    typedef Frame<OperatorCreatePairStaticArray<1u >, ParticleDescriptionBorder> ParticleTypeBorder;


    typedef ParticleType FrameType;
    typedef SuperCell<FrameType> SuperCellType;

private:

    /*this is only for internel calculations*/
    enum
    {
        SizeOfOneBorderElement = (sizeof (ParticleTypeBorder) + sizeof (PopPushType))
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
    superCellSize(superCellSize), gridSize(layout), framesExchanges(NULL)
    {

        exchangeMemoryIndexer = new GridBuffer<PopPushType, DIM1 > (DataSpace<DIM1 > (0));
        framesExchanges = new GridBuffer< ParticleType, DIM1, ParticleTypeBorder > (DataSpace<DIM1 > (0));

        //std::cout << "size: " << sizeof (ParticleType) << " " << sizeof (ParticleTypeBorder) << std::endl;
        DataSpace<DIM> superCellsCount = gridSize / superCellSize;

        superCells = new GridBuffer<SuperCellType, DIM > (superCellsCount);

    }

    void createParticleBuffer()
    {
        reset();
    }

    /**
     * Destructor.
     */
    virtual ~ParticlesBuffer()
    {
        __delete(superCells);
        __delete(framesExchanges);
        __delete(exchangeMemoryIndexer);
    }

    /**
     * Resets all internal buffers.
     */
    void reset()
    {

        superCells->getDeviceBuffer().setValue(SuperCellType ());
        superCells->getHostBuffer().setValue(SuperCellType ());
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
     * Returns a ParticlesBox for device frame data.
     *
     * @return device frames ParticlesBox
     */
    ParticlesBox<ParticleType, DIM> getDeviceParticleBox()
    {

        return ParticlesBox<ParticleType, DIM > (
                                                 superCells->getDeviceBuffer().getDataBox());
    }

    /**
     * Returns a ParticlesBox for host frame data.
     *
     * @return host frames ParticlesBox
     */
    ParticlesBox<ParticleType, DIM> getHostParticleBox(int64_t memoryOffset)
    {

        return ParticlesBox<ParticleType, DIM > (
                                                 superCells->getHostBuffer().getDataBox(),
                                                 memoryOffset
                                                );
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

    void deviceToHost()
    {
        superCells->deviceToHost();
    }


private:
    GridBuffer<PopPushType, DIM1> *exchangeMemoryIndexer;

    GridBuffer<SuperCellType, DIM> *superCells;
    /*gridbuffer for hold borderFrames, we need a own buffer to create first exchanges without core momory*/
    GridBuffer< ParticleType, DIM1, ParticleTypeBorder> *framesExchanges;

    DataSpace<DIM> superCellSize;
    DataSpace<DIM> gridSize;

};
}
