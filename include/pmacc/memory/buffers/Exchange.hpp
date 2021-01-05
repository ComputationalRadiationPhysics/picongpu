/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz
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

#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"

namespace pmacc
{
    /**
     * Interface for a DIM-dimensional buffer used for data exchange.
     *
     * Exchange defines an interface for exchanging data between hosts.
     * Equally sized buffers are created on the device as well as on the host.
     * Exchange buffers may use parts of existing GridBuffer or
     * be used as dedicated memory.
     *
     * @tparam TYPE the datatype for internal buffers
     * @tparam DIM the dimension of the internal buffers
     */
    template<class TYPE, unsigned DIM>
    class Exchange
    {
    public:
        /**
         * Returns the exchange buffer on the device.
         *
         * @return Exchange buffer on device
         */
        virtual DeviceBuffer<TYPE, DIM>& getDeviceBuffer() = 0;

        /**
         * Returns the exchange buffer on the host.
         *
         * @return Exchange buffer on host
         */
        virtual HostBuffer<TYPE, DIM>& getHostBuffer() = 0;

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
        virtual Buffer<TYPE, DIM>* getCommunicationBuffer() = 0;

        virtual bool hasDeviceDoubleBuffer() = 0;

        virtual DeviceBuffer<TYPE, DIM>& getDeviceDoubleBuffer() = 0;

    protected:
        Exchange(uint32_t extype, uint32_t tag) : exchange(extype), communicationTag(tag)
        {
        }

        uint32_t exchange;
        uint32_t communicationTag;
    };

} // namespace pmacc
