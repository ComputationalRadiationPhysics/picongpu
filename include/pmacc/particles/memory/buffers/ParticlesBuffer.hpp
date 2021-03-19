/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "pmacc/particles/frame_types.hpp"
#include "pmacc/memory/buffers/GridBuffer.hpp"
#include "pmacc/particles/memory/boxes/ParticlesBox.hpp"
#include "pmacc/dimensions/GridLayout.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/particles/memory/buffers/StackExchangeBuffer.hpp"
#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/particles/memory/dataTypes/SuperCell.hpp"

#include "pmacc/math/Vector.hpp"

#include "pmacc/particles/boostExtension/InheritGenerators.hpp"
#include "pmacc/meta/conversion/MakeSeq.hpp"


#include <boost/mpl/vector.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/back_inserter.hpp>

#include "pmacc/particles/memory/frames/Frame.hpp"
#include "pmacc/particles/Identifier.hpp"
#include "pmacc/particles/memory/dataTypes/StaticArray.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include "pmacc/particles/ParticleDescription.hpp"
#include "pmacc/particles/memory/dataTypes/ListPointer.hpp"
#include "pmacc/traits/GetUniqueTypeId.hpp"

#include <memory>

namespace pmacc
{
    /**
     * Describes DIM-dimensional buffer for particles data on the host.
     *
     * @tParam T_ParticleDescription Object which describe a frame @see ParticleDescription.hpp
     * @tparam SuperCellSize_ TVec which descripe size of a superce
     * @tparam DIM dimension of the buffer (1-3)
     */
    template<typename T_ParticleDescription, class SuperCellSize_, typename T_DeviceHeap, unsigned DIM>
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
                typedef bmpl::
                    pair<X, StaticArray<typename traits::Resolve<X>::type::type, bmpl::integral_c<uint32_t, T_size>>>
                        type;
            };
        };

        /** type of the border frame management object
         *
         * contains:
         *   - superCell position of the border frames inside a given range
         *   - start position inside the exchange stack for frames
         *   - number of frames corresponding to the superCell position
         */
        typedef ExchangeMemoryIndex<vint_t, DIM - 1> BorderFrameIndex;

        typedef SuperCellSize_ SuperCellSize;

        typedef typename MakeSeq<typename T_ParticleDescription::ValueTypeSeq, localCellIdx, multiMask>::type
            ParticleAttributeList;

        typedef typename MakeSeq<typename T_ParticleDescription::ValueTypeSeq, localCellIdx>::type
            ParticleAttributeListBorder;

        typedef typename ReplaceValueTypeSeq<T_ParticleDescription, ParticleAttributeList>::type
            FrameDescriptionWithManagementAttributes;

        /** double linked list pointer */
        typedef typename MakeSeq<PreviousFramePtr<>, NextFramePtr<>>::type LinkedListPointer;

        /* extent particle description with pointer to a frame*/
        typedef typename ReplaceFrameExtensionSeq<FrameDescriptionWithManagementAttributes, LinkedListPointer>::type
            FrameDescription;

        /** frame definition
         *
         * a group of particles is stored as frame
         */
        typedef Frame<
            OperatorCreatePairStaticArray<pmacc::math::CT::volume<SuperCellSize>::type::value>,
            FrameDescription>
            FrameType;

        typedef typename ReplaceValueTypeSeq<T_ParticleDescription, ParticleAttributeListBorder>::type
            FrameDescriptionBorder;

        /** frame which is used to communicate particles to neighbors
         *
         * - each frame contains only one particle
         * - local administration attributes of a particle are removed
         */
        typedef Frame<OperatorCreatePairStaticArray<1u>, FrameDescriptionBorder> FrameTypeBorder;

        typedef SuperCell<FrameType> SuperCellType;

        typedef T_DeviceHeap DeviceHeap;
        /* Type of the particle box which particle buffer create */
        typedef ParticlesBox<FrameType, typename DeviceHeap::AllocatorHandle, DIM> ParticlesBoxType;

    private:
        /* this enum is used only for internal calculations */
        enum
        {
            SizeOfOneBorderElement = (sizeof(FrameTypeBorder) + sizeof(BorderFrameIndex))
        };

    public:
        /**
         * Constructor.
         *
         * @param deviceHeap device heap memory allocator
         * @param layout number of cell per dimension
         * @param superCellSize size of one super cell
         * @param gpuMemory how many memory on device is used for this instance (in byte)
         */
        ParticlesBuffer(
            const std::shared_ptr<DeviceHeap>& deviceHeap,
            DataSpace<DIM> layout,
            DataSpace<DIM> superCellSize)
            : m_deviceHeap(deviceHeap)
            , superCellSize(superCellSize)
            , gridSize(layout)
            , framesExchanges(nullptr)
        {
            exchangeMemoryIndexer = new GridBuffer<BorderFrameIndex, DIM1>(DataSpace<DIM1>(0));
            framesExchanges = new GridBuffer<FrameType, DIM1, FrameTypeBorder>(DataSpace<DIM1>(0));

            DataSpace<DIM> superCellsCount = gridSize / superCellSize;

            superCells = new GridBuffer<SuperCellType, DIM>(superCellsCount);

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
            superCells->getDeviceBuffer().setValue(SuperCellType());
            superCells->getHostBuffer().setValue(SuperCellType());
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

            framesExchanges
                ->addExchangeBuffer(receive, DataSpace<DIM1>(numFrameTypeBorders), communicationTag, true, false);

            /* Generate a new tag from this type
             *
             * The tag is the same each time this method is called (per instantiation of this template).
             * Here it is fine, as there is the only instance object, and
             * exchangeMemoryIndexer->addExchangeBuffer() requires the same tag on each call
             */
            auto const newTag = traits::GetUniqueTypeId<ParticlesBuffer, uint32_t>::uid();
            exchangeMemoryIndexer
                ->addExchangeBuffer(receive, DataSpace<DIM1>(numFrameTypeBorders), newTag, true, false);
        }

        /**
         * Returns a ParticlesBox for device frame data.
         *
         * @return device frames ParticlesBox
         */
        ParticlesBoxType getDeviceParticleBox()
        {
            return ParticlesBoxType(superCells->getDeviceBuffer().getDataBox(), m_deviceHeap->getAllocatorHandle());
        }

        /**
         * Returns a ParticlesBox for host frame data.
         *
         * @return host frames ParticlesBox
         */
        ParticlesBoxType getHostParticleBox(int64_t memoryOffset)
        {
            return ParticlesBoxType(
                superCells->getHostBuffer().getDataBox(),
                m_deviceHeap->getAllocatorHandle(),
                memoryOffset);
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

        StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1> getSendExchangeStack(uint32_t ex)
        {
            return StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1>(
                framesExchanges->getSendExchange(ex),
                exchangeMemoryIndexer->getSendExchange(ex));
        }

        StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1> getReceiveExchangeStack(uint32_t ex)
        {
            return StackExchangeBuffer<FrameTypeBorder, BorderFrameIndex, DIM - 1>(
                framesExchanges->getReceiveExchange(ex),
                exchangeMemoryIndexer->getReceiveExchange(ex));
        }

        /**
         * Starts sync data from own device buffer to neighbor device buffer.
         *
         * GridBuffer
         *
         */
        EventTask asyncCommunication(EventTask serialEvent)
        {
            return framesExchanges->asyncCommunication(serialEvent)
                + exchangeMemoryIndexer->asyncCommunication(serialEvent);
        }

        EventTask asyncSendParticles(EventTask serialEvent, uint32_t ex)
        {
            /* store each gpu-free event separately to avoid race conditions */
            EventTask framesExchangesGPUEvent;
            EventTask exchangeMemoryIndexerGPUEvent;
            EventTask returnEvent
                = framesExchanges->asyncSend(serialEvent, ex) + exchangeMemoryIndexer->asyncSend(serialEvent, ex);

            return returnEvent;
        }

        EventTask asyncReceiveParticles(EventTask serialEvent, uint32_t ex)
        {
            return framesExchanges->asyncReceive(serialEvent, ex)
                + exchangeMemoryIndexer->asyncReceive(serialEvent, ex);
        }

        /**
         * Returns number of supercells in each dimension.
         *
         * @return number of supercells
         */
        DataSpace<DIM> getSuperCellsCount()
        {
            PMACC_ASSERT(superCells != nullptr);
            return superCells->getGridLayout().getDataSpace();
        }

        /**
         * Returns number of supercells in each dimension.
         *
         * @return number of supercells
         */
        GridLayout<DIM> getSuperCellsLayout()
        {
            PMACC_ASSERT(superCells != nullptr);
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
        GridBuffer<BorderFrameIndex, DIM1>* exchangeMemoryIndexer;

        GridBuffer<SuperCellType, DIM>* superCells;
        /*GridBuffer for hold borderFrames, we need a own buffer to create first exchanges without core memory*/
        GridBuffer<FrameType, DIM1, FrameTypeBorder>* framesExchanges;

        DataSpace<DIM> superCellSize;
        DataSpace<DIM> gridSize;
        std::shared_ptr<DeviceHeap> m_deviceHeap;
    };
} // namespace pmacc
