/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/dimensions/GridLayout.hpp"
#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/mappings/simulation/EnvironmentController.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/memory/buffers/ExchangeIntern.hpp"
#include "pmacc/memory/buffers/HostBufferIntern.hpp"
#include "pmacc/memory/buffers/DeviceBufferIntern.hpp"
#include "pmacc/memory/buffers/GridBuffer.hpp"
#include "pmacc/memory/boxes/MultiBox.hpp"
#include "pmacc/assert.hpp"

#include <algorithm>

namespace pmacc
{
    template<typename Type_, uint32_t communicationTag_ = 0, bool sizeOnDevice_ = false>
    struct TypeDescriptionElement
    {
        typedef Type_ Type;
        static constexpr uint32_t communicationTag = communicationTag_;
        static constexpr bool sizeOnDevice = sizeOnDevice_;
    };

    /**
     * GridBuffer represents a DIM-dimensional buffer which exists on the host as well as on the device.
     *
     * GridBuffer combines a HostBuffer and a DeviceBuffer with equal sizes.
     * Additionally, it allows sending data from and receiving data to these buffers.
     * Buffers consist of core data which may be surrounded by border data.
     *
     * @tparam Type_ datatype for internal Host- and DeviceBuffer
     * @tparam DIM dimension of the buffers
     * @tparam BufferNames a class with a enum with the name "Names" and member with the name "Count" with number of
     * elements in Names etc.: struct Mem
     *  {
     *    enum Names{VALUE1,VALUE2};
     *    static constexpr uint32_t Count=2;
     *  };
     * @tparam BORDERTYPE optional type for border data in the buffers. TYPE is used by default.
     */
    template<typename Type_, unsigned DIM, class BufferNames, class BORDERTYPE = Type_>
    class MultiGridBuffer
    {
    public:
        typedef Type_ Type;
        typedef DataBox<MultiBox<Type, DIM>> DataBoxType;
        typedef GridBuffer<Type, DIM> GridBufferType;
        typedef typename BufferNames::Names NameType;

        /**
         * Constructor.
         *
         * @param gridLayout layout of the buffers, including border-cells
         * @param firstCommunicationTag optional value which can be used to tag ('name') this buffer in communications
         * @param sizeOnDevice if true, size information exists on device, too.
         */
        MultiGridBuffer(const GridLayout<DIM>& gridLayout, bool sizeOnDevice = false)
            : blobDeviceBuffer(nullptr)
            , blobHostBuffer(nullptr)
        {
            init(gridLayout, sizeOnDevice);
        }

        /**
         * Constructor.
         *
         * @param dataSpace DataSpace representing buffer size without border-cells
         * @param firstCommunicationTag optional value which can be used to tag ('name') this buffer in communications
         * @param sizeOnDevice if true, size information exists on device, too.
         */
        MultiGridBuffer(DataSpace<DIM>& dataSpace, bool sizeOnDevice = false)
            : blobDeviceBuffer(nullptr)
            , blobHostBuffer(nullptr)
        {
            init(GridLayout<DIM>(dataSpace), sizeOnDevice);
        }

        /**
         * Add Exchange in MultiGridBuffer memory space.
         *
         * An Exchange is added to this MultiGridBuffer. The exchange buffers use
         * the same memory as this MultiGridBuffer.
         *
         * @param dataPlace place where received data are stored [GUARD | BORDER]
         *        if dataPlace=GUARD than copy other BORDER to my GUARD
         *        if dataPlace=BORDER than copy other GUARD to my BORDER
         * @param receive a Mask which describes the directions for the exchange
         * @param guardingCells number of guarding cells in each dimension
         * @param firstCommunicationTag a object unique number to connect same objects from different nodes
         * (MultiGridBuffer reserves all tags from [firstCommunicationTag;firstCommunicationTag+BufferNames::Count]
         * @param sizeOnDevice if true, internal buffers have their size information on the device, too
         */
        void addExchange(
            uint32_t dataPlace,
            const Mask& receive,
            DataSpace<DIM> guardingCells,
            uint32_t firstCommunicationTag,
            bool sizeOnDevice = false)
        {
            for(uint32_t i = 0; i < BufferNames::Count; ++i)
            {
                getGridBuffer(static_cast<NameType>(i))
                    .addExchange(dataPlace, receive, guardingCells, firstCommunicationTag + i, sizeOnDevice);
            }
        }

        /**
         * Destructor.
         */
        virtual ~MultiGridBuffer()
        {
            for(uint32_t i = 0; i < BufferNames::Count; ++i)
            {
                __delete(gridBuffers[i]);
            }
            __delete(blobDeviceBuffer);
            __delete(blobHostBuffer);
        }

        /**
         * Resets both internal buffers.
         *
         * See DeviceBuffer::reset and HostBuffer::reset for details.
         *
         * @param preserveData determines if data on internal buffers should not be erased
         */
        void reset(bool preserveData = true)
        {
            for(uint32_t i = 0; i < BufferNames::Count; ++i)
            {
                getGridBuffer(static_cast<NameType>(i)).reset(preserveData);
            }
        }

        /**
         * Starts sync data from own device buffer to neighboring device buffer.
         *
         * Asynchronously starts synchronization of data from internal DeviceBuffer using added
         * Exchange buffers.
         *
         */
        EventTask asyncCommunication(EventTask serialEvent)
        {
            EventTask ev;

            for(uint32_t i = 0; i < BufferNames::Count; ++i)
            {
                ev += getGridBuffer(static_cast<NameType>(i)).asyncCommunication(serialEvent);
            }
            return ev;
        }

        /**
         * Starts sync data from own device buffer to neighboring device buffer.
         *
         * Asynchronously starts synchronization of data from internal DeviceBuffer using added
         * Exchange buffers.
         * This operation runs sequentially to other code but uses asynchronous operations internally.
         *
         */
        EventTask communication()
        {
            EventTask ev;
            EventTask serialEvent = __getTransactionEvent();

            for(uint32_t i = 0; i < BufferNames::Count; ++i)
            {
                ev += getGridBuffer(static_cast<NameType>(i)).asyncCommunication(serialEvent);
            }
            __setTransactionEvent(ev);
            return ev;
        }

        /**
         * Asynchronously copies data from internal host to internal device buffer.
         *
         */
        void hostToDevice()
        {
            for(uint32_t i = 0; i < BufferNames::Count; ++i)
            {
                getGridBuffer(static_cast<NameType>(i)).hostToDevice();
            }
        }

        /**
         * Asynchronously copies data from internal device to internal host buffer.
         */
        void deviceToHost()
        {
            for(uint32_t i = 0; i < BufferNames::Count; ++i)
            {
                getGridBuffer(static_cast<NameType>(i)).deviceToHost();
            }
        }

        GridBuffer<Type, DIM>& getGridBuffer(typename BufferNames::Names name)
        {
            PMACC_ASSERT(name >= 0 && name < BufferNames::Count);
            return *gridBuffers[name];
        }

        DataBoxType getHostDataBox()
        {
            __startOperation(ITask::TASK_HOST);
            return DataBoxType(MultiBox<Type, DIM>(
                getGridBuffer(static_cast<NameType>(0)).getHostBuffer().getBasePointer(),
                DataSpace<DIM>(),
                getGridBuffer(static_cast<NameType>(0)).getHostBuffer().getPhysicalMemorySize(),
                getGridBuffer(static_cast<NameType>(0)).getHostBuffer().getPhysicalMemorySize().x() * sizeof(Type)));
        }

        DataBoxType getDeviceDataBox()
        {
            __startOperation(ITask::TASK_DEVICE);
            return DataBoxType(MultiBox<Type, DIM>(
                getGridBuffer(static_cast<NameType>(0)).getDeviceBuffer().getBasePointer(),
                getGridBuffer(static_cast<NameType>(0)).getDeviceBuffer().getOffset(),
                getGridBuffer(static_cast<NameType>(0)).getDeviceBuffer().getPhysicalMemorySize(),
                getGridBuffer(static_cast<NameType>(0)).getDeviceBuffer().getCudaPitched().pitch));
        }

    private:
        void init(GridLayout<DIM> gridLayout, bool sizeOnDevice)
        {
            DataSpace<DIM> blobOffset;
            blobOffset[DIM - 1] = gridLayout.getDataSpace()[DIM - 1];

            DataSpace<DIM> blobSize = gridLayout.getDataSpace() + blobOffset * (BufferNames::Count - 1);

            blobDeviceBuffer = new DeviceBufferIntern<Type_, DIM>(blobSize, false);
            blobHostBuffer = new HostBufferIntern<Type_, DIM>(blobSize);

            for(uint32_t i = 0; i < BufferNames::Count; ++i)
            {
                DataSpace<DIM> offset = blobOffset * i;
                gridBuffers[i] = new GridBuffer<Type, DIM>(
                    *blobHostBuffer,
                    offset,
                    *blobDeviceBuffer,
                    offset,
                    gridLayout,
                    sizeOnDevice);
            }
        }


    protected:
        DeviceBufferIntern<Type, DIM>* blobDeviceBuffer;
        HostBufferIntern<Type, DIM>* blobHostBuffer;
        GridBufferType* gridBuffers[BufferNames::Count];
    };
} // namespace pmacc
