/**
 * Copyright 2013-2017 Axel Huebl, Felix Schmitt, Rene Widera, Benjamin Worpitz
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
#include "particles/memory/dataTypes/ListPointer.hpp"


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
    template< uint32_t T_size >
    struct OperatorCreatePairStaticArray
    {

        template<typename X>
        struct apply
        {
            typedef bmpl::pair<
                X,
                StaticArray<
                    typename traits::Resolve<X>::type::type,
                    bmpl::integral_c<uint32_t, T_size>
                >
            > type;
        };
    };

    /** type of the border frame management object
     *
     * contains:
     *   - superCell position of the border frames inside a given range
     *   - start position inside the exchange stack for frames
     *   - number of frames corresponding to the superCell position
     */
    typedef ExchangeMemoryIndex<
        vint_t,
        DIM - 1
    > BorderFrameIndex;

    typedef SuperCellSize_ SuperCellSize;

    typedef typename MakeSeq<
        typename T_ParticleDescription::ValueTypeSeq,
        localCellIdx,
        multiMask,
        inCellOffset
    >::type ParticleAttributeList;

    typedef typename MakeSeq<
        typename T_ParticleDescription::ValueTypeSeq,
        localCellIdx
    >::type ParticleAttributeListBorder;

    typedef
    typename ReplaceValueTypeSeq<
        T_ParticleDescription,
        ParticleAttributeList
    >::type FrameDescriptionWithManagementAttributes;

    /** double linked list pointer */
    typedef
    typename MakeSeq<
        PreviousFramePtr<>,
        NextFramePtr<>
    >::type LinkedListPointer;

    /* extent particle description with pointer to a frame*/
    typedef typename ReplaceFrameExtensionSeq<
        FrameDescriptionWithManagementAttributes,
        LinkedListPointer
    >::type FrameDescription;

    /** frame definition
     *
     * a group of particles is stored as frame
     */
    typedef Frame<
        OperatorCreatePairStaticArray<
            PMacc::math::CT::volume< SuperCellSize >::type::value
        >,
        FrameDescription
    > FrameType;

    typedef typename ReplaceValueTypeSeq<
        T_ParticleDescription,
        ParticleAttributeListBorder
    >::type FrameDescriptionBorder;

    /** frame which is used to communicate particles to neighbors
     *
     * - each frame contains only one particle
     * - local administration attributes of a particle are removed
     */
    typedef Frame<
        OperatorCreatePairStaticArray< 1u >,
        FrameDescriptionBorder
    > FrameTypeBorder;

    typedef SuperCell<FrameType> SuperCellType;

private:

    /* this enum is used only for internal calculations */
    enum
    {
        SizeOfOneBorderElement = (sizeof (FrameTypeBorder) + sizeof (BorderFrameIndex))
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

        exchangeMemoryIndexer = new GridBuffer<BorderFrameIndex, DIM1 > (DataSpace<DIM1 > (0));
        framesExchanges = new GridBuffer< FrameType, DIM1, FrameTypeBorder > (DataSpace<DIM1 > (0));

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

        size_t numFrameTypeBorders = usedMemory / SizeOfOneBorderElement;

        framesExchanges->addExchangeBuffer(receive, DataSpace<DIM1 > (numFrameTypeBorders), communicationTag, true, false);

        exchangeMemoryIndexer->addExchangeBuffer(receive, DataSpace<DIM1 > (numFrameTypeBorders), communicationTag | (1u << (20 - 5)), true, false);
    }

    /**
     * Returns a ParticlesBox for device frame data.
     *
     * @return device frames ParticlesBox
     */
    ParticlesBox<FrameType, DIM> getDeviceParticleBox()
    {

        return ParticlesBox<FrameType, DIM > (
                                                 superCells->getDeviceBuffer().getDataBox());
    }

    /**
     * Returns a ParticlesBox for host frame data.
     *
     * @return host frames ParticlesBox
     */
    ParticlesBox<FrameType, DIM> getHostParticleBox(int64_t memoryOffset)
    {

        return ParticlesBox<FrameType, DIM > (
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

    StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1 > getSendExchangeStack(uint32_t ex)
    {

        return StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1 >
            (framesExchanges->getSendExchange(ex), exchangeMemoryIndexer->getSendExchange(ex));
    }

    StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1 > getReceiveExchangeStack(uint32_t ex)
    {

        return StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1 >
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

    EventTask asyncSendParticles(EventTask serialEvent, uint32_t ex)
    {
        /* store each gpu-free event separately to avoid race conditions */
        EventTask framesExchangesGPUEvent;
        EventTask exchangeMemoryIndexerGPUEvent;
        EventTask returnEvent = framesExchanges->asyncSend(serialEvent, ex) +
            exchangeMemoryIndexer->asyncSend(serialEvent, ex);

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

        PMACC_ASSERT(superCells != NULL);
        return superCells->getGridLayout().getDataSpace();
    }

    /**
     * Returns number of supercells in each dimension.
     *
     * @return number of supercells
     */
    GridLayout<DIM> getSuperCellsLayout()
    {

        PMACC_ASSERT(superCells != NULL);
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
    GridBuffer<BorderFrameIndex, DIM1> *exchangeMemoryIndexer;

    GridBuffer<SuperCellType, DIM> *superCells;
    /*GridBuffer for hold borderFrames, we need a own buffer to create first exchanges without core memory*/
    GridBuffer< FrameType, DIM1, FrameTypeBorder> *framesExchanges;

    DataSpace<DIM> superCellSize;
    DataSpace<DIM> gridSize;

};
}
