/**
 * Copyright 2013-2015 Rene Widera, Benjamin Worpitz
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
#include "eventSystem/EventSystem.hpp"
#include "mappings/simulation/EnvironmentController.hpp"
#include "memory/dataTypes/Mask.hpp"
#include "memory/buffers/ExchangeIntern.hpp"
#include "memory/buffers/HostBufferIntern.hpp"
#include "memory/buffers/DeviceBufferIntern.hpp"

#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <set>

namespace PMacc
{
namespace privateGridBuffer
{

class UniquTag
{
public:

    static UniquTag& getInstance()
    {
        static UniquTag instance;
        return instance;
    }

    bool isTagUniqu(uint32_t tag)
    {
        bool isUniqu = tags.find(tag) == tags.end();
        if (isUniqu)
            tags.insert(tag);
        return isUniqu;
    }
private:

    UniquTag()
    {
    }

    /**
     * Constructor
     */
    UniquTag(const UniquTag&)
    {

    }

    std::set<uint32_t> tags;
};

}//end namespace privateGridBuffer

/**
 * GridBuffer represents a DIM-dimensional buffer which exists on the host as well as on the device.
 *
 * GridBuffer combines a HostBuffer and a DeviceBuffer with equal sizes.
 * Additionally, it allows sending data from and receiving data to these buffers.
 * Buffers consist of core data which may be surrounded by border data.
 *
 * @tparam TYPE datatype for internal Host- and DeviceBuffer
 * @tparam DIM dimension of the buffers
 * @tparam BORDERTYPE optional type for border data in the buffers. TYPE is used by default.
 */
template <class TYPE, unsigned DIM, class BORDERTYPE = TYPE>
class GridBuffer
{
public:

    typedef DataBox<PitchedBox<TYPE, DIM> > DataBoxType;

    /**
     * Constructor.
     *
     * @param gridLayout layout of the buffers, including border-cells
     * @param sizeOnDevice if true, size information exists on device, too.
     */
    GridBuffer(const GridLayout<DIM>& gridLayout, bool sizeOnDevice = false) :
    gridLayout(gridLayout),
    hasOneExchange(false),
    maxExchange(0)
    {
        init(sizeOnDevice);
    }

    /**
     * Constructor.
     *
     * @param dataSpace DataSpace representing buffer size without border-cells
     * @param sizeOnDevice if true, internal buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     */
    GridBuffer(DataSpace<DIM>& dataSpace, bool sizeOnDevice = false) :
    gridLayout(GridLayout<DIM>(dataSpace)),
    hasOneExchange(false),
    maxExchange(0)
    {
        init(sizeOnDevice);
    }

    /**
     * Constructor.
     *
     * @param otherDeviceBuffer DeviceBuffer which should be used instead of creating own DeviceBuffer
     * @param gridLayout layout of the buffers, including border-cells
     * @param sizeOnDevice if true, internal buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     */
    GridBuffer(DeviceBuffer<TYPE, DIM>& otherDeviceBuffer, GridLayout<DIM> gridLayout, bool sizeOnDevice = false) :
    gridLayout(gridLayout),
    hasOneExchange(false),
    maxExchange(0)
    {
        init(sizeOnDevice, false);
        this->deviceBuffer = new DeviceBufferIntern<TYPE, DIM >
            (otherDeviceBuffer,
             this->gridLayout.getDataSpace(),
             DataSpace<DIM > (),
             sizeOnDevice);
    }

    GridBuffer(
               HostBuffer<TYPE, DIM>& otherHostBuffer,
               DataSpace<DIM > offsetHost,
               DeviceBuffer<TYPE, DIM>& otherDeviceBuffer,
               DataSpace<DIM > offsetDevice,
               GridLayout<DIM> gridLayout,
               bool sizeOnDevice = false) :
    gridLayout(gridLayout),
    hasOneExchange(false),
    maxExchange(0)
    {
        init(sizeOnDevice, false, false);
        this->deviceBuffer = new DeviceBufferIntern<TYPE, DIM >
            (otherDeviceBuffer,
             this->gridLayout.getDataSpace(),
             offsetDevice, sizeOnDevice);
        this->hostBuffer = new HostBufferIntern<TYPE, DIM >
            (*((HostBufferIntern<TYPE, DIM>*) & otherHostBuffer),
             this->gridLayout.getDataSpace(),
             offsetHost);
    }

    /**
     * Destructor.
     */
    virtual ~GridBuffer()
    {
        for (uint32_t i = 0; i < 27; ++i)
        {
            __delete(sendExchanges[i]);
            __delete(receiveExchanges[i]);
        }

        __delete(hostBuffer);
        __delete(deviceBuffer);
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
        deviceBuffer->reset(preserveData);
        hostBuffer->reset(preserveData);
    }

    /**
     * Add Exchange in GridBuffer memory space.
     *
     * An Exchange is added to this GridBuffer. The exchange buffers use
     * the same memory as this GridBuffer.
     *
     * @param dataPlace place where received data is stored [GUARD | BORDER]
     *        if dataPlace=GUARD than copy other BORDER to my GUARD
     *        if dataPlace=BORDER than copy other GUARD to my BORDER
     * @param receive a Mask which describes the directions for the exchange
     * @param guardingCells number of guarding cells in each dimension
     * @param communicationTag unique tag/id for communication
     * @param sizeOnDeviceSend if true, internal send buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     * @param sizeOnDeviceReceive if true, internal receive buffers must store their
     *        size additionally on the device
     */
    void addExchange(uint32_t dataPlace, const Mask &receive, DataSpace<DIM> guardingCells, uint32_t communicationTag, bool sizeOnDeviceSend, bool sizeOnDeviceReceive )
    {

        if (hasOneExchange && (communicationTag != lastUsedCommunicationTag))
            throw std::runtime_error("It is not allowed to give the same GridBuffer different communicationTags");

        lastUsedCommunicationTag = communicationTag;

        receiveMask = receiveMask + receive;
        sendMask = this->receiveMask.getMirroredMask();
        Mask send = receive.getMirroredMask();



        for (uint32_t ex = 1; ex< -12 * (int) DIM + 6 * (int) DIM * (int) DIM + 9; ++ex)
        {
            if (send.isSet(ex))
            {
                uint32_t uniqCommunicationTag = (communicationTag << 5) | ex;

                if (!hasOneExchange && !privateGridBuffer::UniquTag::getInstance().isTagUniqu(uniqCommunicationTag))
                {
                    std::stringstream message;
                    message << "unique exchange communication tag ("
                        << uniqCommunicationTag << ") witch is created from communicationTag ("
                        << communicationTag << ") allready used for other gridbuffer exchange";
                    throw std::runtime_error(message.str());
                }
                hasOneExchange = true;

                if (sendExchanges[ex] != NULL)
                {
                    throw std::runtime_error("Exchange already added!");
                }
                //std::cout<<"Add Exchange: send="<<ex<<" receive="<<Mask::getMirroredExchangeType((ExchangeType)ex)<<std::endl;
                maxExchange = std::max(maxExchange, ex + 1u);
                sendExchanges[ex] = new ExchangeIntern<BORDERTYPE, DIM > (*deviceBuffer, gridLayout, guardingCells,
                                                                          (ExchangeType) ex, uniqCommunicationTag,
                                                                          dataPlace == GUARD ? BORDER : GUARD, sizeOnDeviceSend);
                ExchangeType recvex = Mask::getMirroredExchangeType(ex);
                maxExchange = std::max(maxExchange, recvex + 1u);
                receiveExchanges[recvex] =
                    new ExchangeIntern<BORDERTYPE, DIM > (
                                                          *deviceBuffer,
                                                          gridLayout,
                                                          guardingCells,
                                                          recvex,
                                                          uniqCommunicationTag,
                                                          dataPlace == GUARD ? GUARD : BORDER,
                                                          sizeOnDeviceReceive);
            }
        }
    }

    /**
     * Add Exchange in GridBuffer memory space.
     *
     * An Exchange is added to this GridBuffer. The exchange buffers use
     * the same memory as this GridBuffer.
     *
     * @param dataPlace place where received data is stored [GUARD | BORDER]
     *        if dataPlace=GUARD than copy other BORDER to my GUARD
     *        if dataPlace=BORDER than copy other GUARD to my BORDER
     * @param receive a Mask which describes the directions for the exchange
     * @param guardingCells number of guarding cells in each dimension
     * @param communicationTag unique tag/id for communication
     * @param sizeOnDevice if true, internal buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     */
    void addExchange(uint32_t dataPlace, const Mask &receive, DataSpace<DIM> guardingCells, uint32_t communicationTag, bool sizeOnDevice = false)
    {
        addExchange( dataPlace, receive, guardingCells, communicationTag, sizeOnDevice, sizeOnDevice );
    }

    /**
     * Add Exchange in dedicated memory space.
     *
     * An Exchange is added to this GridBuffer. The exchange buffers use
     * the their own memory instead of using the GridBuffer's memory space.
     *
     * @param receive a Mask which describes the directions for the exchange
     * @param dataSpace size of the newly created exchange buffer in each dimension
     * @param communicationTag unique tag/id for communication
     * @param sizeOnDeviceSend if true, internal send buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     * @param sizeOnDeviceReceive if true, internal receive buffers must store their
     *        size additionally on the device
     */
    void addExchangeBuffer(const Mask &receive, const DataSpace<DIM> &dataSpace, uint32_t communicationTag, bool sizeOnDeviceSend, bool sizeOnDeviceReceive )
    {

        if (hasOneExchange && (communicationTag != lastUsedCommunicationTag))
            throw std::runtime_error("It is not allowed to give the same GridBuffer different communicationTags");
        lastUsedCommunicationTag = communicationTag;


        /*don't create buffer with 0 (zero) elements*/
        if (dataSpace.productOfComponents() != 0)
        {
            receiveMask = receiveMask + receive;
            sendMask = this->receiveMask.getMirroredMask();
            Mask send = receive.getMirroredMask();
            for (uint32_t ex = 1; ex < 27; ++ex)
            {
                if (send.isSet(ex))
                {
                    uint32_t uniqCommunicationTag = (communicationTag << 5) | ex;
                    if (!hasOneExchange && !privateGridBuffer::UniquTag::getInstance().isTagUniqu(uniqCommunicationTag))
                    {
                        std::stringstream message;
                        message << "unique exchange communication tag ("
                            << uniqCommunicationTag << ") witch is created from communicationTag ("
                            << communicationTag << ") allready used for other gridbuffer exchange";
                        throw std::runtime_error(message.str());
                    }
                    hasOneExchange = true;

                    if (sendExchanges[ex] != NULL)
                    {
                        throw std::runtime_error("Exchange already added!");
                    }

                    //GridLayout<DIM> memoryLayout(size);
                    maxExchange = std::max(maxExchange, ex + 1u);
                    sendExchanges[ex] = new ExchangeIntern<BORDERTYPE, DIM > (/*memoryLayout*/ dataSpace,
                                                                              ex, uniqCommunicationTag, sizeOnDeviceSend);

                    ExchangeType recvex = Mask::getMirroredExchangeType(ex);
                    maxExchange = std::max(maxExchange, recvex + 1u);
                    receiveExchanges[recvex] = new ExchangeIntern<BORDERTYPE, DIM > (/*memoryLayout*/ dataSpace,
                                                                                     recvex, uniqCommunicationTag, sizeOnDeviceReceive);
                }
            }
        }
    }

    /**
     * Add Exchange in dedicated memory space.
     *
     * An Exchange is added to this GridBuffer. The exchange buffers use
     * the their own memory instead of using the GridBuffer's memory space.
     *
     * @param receive a Mask which describes the directions for the exchange
     * @param dataSpace size of the newly created exchange buffer in each dimension
     * @param communicationTag unique tag/id for communication
     * @param sizeOnDevice if true, internal buffers must store their
     *        size additionally on the device
     *        (as we keep this information coherent with the host, it influences
     *        performance on host-device copies, but some algorithms on the device
     *        might need to know the size of the buffer)
     */
    void addExchangeBuffer(const Mask &receive, const DataSpace<DIM> &dataSpace, uint32_t communicationTag, bool sizeOnDevice = false )
    {
        addExchangeBuffer( receive, dataSpace, communicationTag, sizeOnDevice, sizeOnDevice );
    }

    /**
     * Returns whether this GridBuffer has an Exchange for sending in ex direction.
     *
     * @param ex exchange direction to query
     * @return true if send exchanges with ex direction exist, otherwise false
     */
    bool hasSendExchange(uint32_t ex) const
    {
        return ( (sendExchanges[ex] != NULL) && (getSendMask().isSet(ex)));
    }

    /**
     * Returns whether this GridBuffer has an Exchange for receiving from ex direction.
     *
     * @param ex exchange direction to query
     * @return true if receive exchanges with ex direction exist, otherwise false
     */
    bool hasReceiveExchange(uint32_t ex) const
    {
        return ( (receiveExchanges[ex] != NULL) && (getReceiveMask().isSet(ex)));
    }

    /**
     * Returns the Exchange for sending data in ex direction.
     *
     * Returns an Exchange which for sending data from
     * this GridBuffer in the direction described by ex.
     *
     * @param ex the direction to query
     * @return the Exchange for sending data
     */
    Exchange<BORDERTYPE, DIM>& getSendExchange(uint32_t ex) const
    {
        return *sendExchanges[ex];
    }

    /**
     * Returns the Exchange for receiving data from ex direction.
     *
     * Returns an Exchange which for receiving data to
     * this GridBuffer from the direction described by ex.
     *
     * @param ex the direction to query
     * @return the Exchange for receiving data
     */
    Exchange<BORDERTYPE, DIM>& getReceiveExchange(uint32_t ex) const
    {
        return *receiveExchanges[ex];
    }

    /**
     * Returns the Mask describing send exchanges
     *
     * @return Mask for send exchanges
     */
    Mask getSendMask() const
    {
        return (Environment<DIM>::get().EnvironmentController().getCommunicationMask() & sendMask);
    }

    /**
     * Returns the Mask describing receive exchanges
     *
     * @return Mask for receive exchanges
     */
    Mask getReceiveMask() const
    {
        return (Environment<DIM>::get().EnvironmentController().getCommunicationMask() & receiveMask);
    }

    /**
     * Returns the internal data buffer on host side
     *
     * @return internal HostBuffer
     */
    HostBuffer<TYPE, DIM>& getHostBuffer() const
    {
        return *(this->hostBuffer);
    }

    /**
     * Returns the internal data buffer on device side
     *
     * @return internal DeviceBuffer
     */
    DeviceBuffer<TYPE, DIM>& getDeviceBuffer() const
    {
        return *(this->deviceBuffer);
    }

    /**
     * Starts sync data from own device buffer to neigbhor device buffer.
     *
     * Asynchronously starts syncronization data from internal DeviceBuffer using added
     * Exchange buffers.
     * This operation runs sequential to other code but intern asyncron
     *
     */
    EventTask communication()
    {
        EventTask ev = this->asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(ev);
        return ev;
    }

    /**
     * Starts sync data from own device buffer to neigbhor device buffer.
     *
     * Asynchronously starts syncronization data from internal DeviceBuffer using added
     * Exchange buffers.
     *
     */
    EventTask asyncCommunication(EventTask serialEvent)
    {
        EventTask evR;
        for (uint32_t i = 0; i < maxExchange; ++i)
        {

            evR += asyncReceive(serialEvent, i);

            ExchangeType sendEx = Mask::getMirroredExchangeType(i);

            EventTask copyEvent;
            asyncSend(serialEvent, sendEx, copyEvent);
            /* add only the copy event, because all work on gpu can run after data is copyed
             */
            evR += copyEvent;

        }
        return evR;
    }

    EventTask asyncSend(EventTask serialEvent, uint32_t sendEx, EventTask &gpuFree)
    {
        if (hasSendExchange(sendEx))
        {
            __startAtomicTransaction(serialEvent + sendEvents[sendEx]);
            sendEvents[sendEx] = sendExchanges[sendEx]->startSend(gpuFree);
            __endTransaction();
            /* add only the copy event, because all work on gpu can run after data is copyed
             */
            return gpuFree;
        }
        return EventTask();
    }

    EventTask asyncReceive(EventTask serialEvent, uint32_t recvEx)
    {
        if (hasReceiveExchange(recvEx))
        {
            __startAtomicTransaction(serialEvent + receiveEvents[recvEx]);
            receiveEvents[recvEx] = receiveExchanges[recvEx]->startReceive();

            __endTransaction();
            return receiveEvents[recvEx];
        }
        return EventTask();
    }

    /**
     * Asynchronously copies data from internal host to internal device buffer.
     *
     */
    void hostToDevice()
    {
        deviceBuffer->copyFrom(*hostBuffer);
    }

    /**
     * Asynchronously copies data from internal device to internal host buffer.
     */
    void deviceToHost()
    {
        hostBuffer->copyFrom(*deviceBuffer);
    }

    /**
     * Returns the GridLayout describing this GridBuffer.
     *
     * @return the layout of this buffer
     */
    GridLayout<DIM> getGridLayout()
    {
        return gridLayout;
    }

private:

    friend class Environment<DIM>;

    void init(bool sizeOnDevice, bool buildDeviceBuffer = true, bool buildHostBuffer = true)
    {
        for (uint32_t i = 0; i < 27; ++i)
        {
            sendExchanges[i] = NULL;
            receiveExchanges[i] = NULL;
            /* fill array with valid empty events to avoid side effects if
             * array is accessed without calling hasExchange() before usage */
            receiveEvents[i] = EventTask();
            sendEvents[i] = EventTask();
        }
        if (buildDeviceBuffer)
        {
            this->deviceBuffer = new DeviceBufferIntern<TYPE, DIM > (gridLayout.getDataSpace(), sizeOnDevice);
        }
        if (buildHostBuffer)
        {
            this->hostBuffer = new HostBufferIntern<TYPE, DIM > (gridLayout.getDataSpace());
        }
    }

protected:


    HostBufferIntern<TYPE, DIM>* hostBuffer;
    DeviceBufferIntern<TYPE, DIM>* deviceBuffer;
    /*if we hase one exchange we not check if communicationtag has used before*/
    bool hasOneExchange;
    uint32_t lastUsedCommunicationTag;
    GridLayout<DIM> gridLayout;

    Mask sendMask;
    Mask receiveMask;

    ExchangeIntern<BORDERTYPE, DIM>* sendExchanges[27];
    ExchangeIntern<BORDERTYPE, DIM>* receiveExchanges[27];
    EventTask receiveEvents[27];
    EventTask sendEvents[27];

    uint32_t maxExchange; //use max exchanges and run over the array is faster as use set from stl
};

}
