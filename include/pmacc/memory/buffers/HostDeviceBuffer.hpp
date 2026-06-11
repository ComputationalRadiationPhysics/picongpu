/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"
#include "pmacc/types.hpp"

#include <memory>
#include <type_traits>

namespace pmacc
{
    /** Buffer that contains a host and device buffer and allows synchronizing those 2 */
    template<typename T_Type, unsigned T_dim>
    class HostDeviceBuffer
    {
        using HostBufferType = HostBuffer<T_Type, T_dim>;
        using DeviceBufferType = DeviceBuffer<T_Type, T_dim>;

    public:
        using ValueType = T_Type;
        using HBuffer = HostBuffer<T_Type, T_dim>;
        using DBuffer = DeviceBuffer<T_Type, T_dim>;
        using DataBoxType = typename HostBufferType::DataBoxType;
        PMACC_CASSERT_MSG(
            DataBoxTypes_must_match,
            std::is_same_v<DataBoxType, typename DeviceBufferType::DataBoxType>);

        /**
         * Constructor that creates the buffers with the given size
         *
         * @param size DataSpace representing buffer size
         * @param sizeOnDevice if true, internal buffers must store their
         *        size additionally on the device
         *        (as we keep this information coherent with the host, it influences
         *        performance on host-device copies, but some algorithms on the device
         *        might need to know the size of the buffer)
         */
        HostDeviceBuffer(DataSpace<T_dim> const& size, bool sizeOnDevice = false);

        /**
         * Constructor that reuses the given device buffer instead of creating an own one.
         * Sizes should match. If size is smaller than the buffer size, then only the part near the origin is used.
         * Passing a size bigger than the buffer is undefined.
         */
        HostDeviceBuffer(DBuffer& otherDeviceBuffer, DataSpace<T_dim> const& size, bool sizeOnDevice = false);

        /**
         * Constructor that reuses the given buffers instead of creating own ones.
         * The data from [offset, offset+size) is used
         * Passing a size bigger than the buffer (minus the offset) is undefined.
         */
        HostDeviceBuffer(
            HBuffer& otherHostBuffer,
            DataSpace<T_dim> const& offsetHost,
            DBuffer& otherDeviceBuffer,
            DataSpace<T_dim> const& offsetDevice,
            GridLayout<T_dim> const size,
            bool sizeOnDevice = false);

        /**
         * Returns the internal data buffer on host side
         *
         * @return internal HBuffer
         */
        HINLINE HBuffer& getHostBuffer() const;

        /**
         * Returns the internal data buffer on device side
         *
         * @return internal DBuffer
         */
        HINLINE DBuffer& getDeviceBuffer() const;

        /**
         * Resets both internal buffers.
         *
         * See DeviceBuffer::reset and HostBuffer::reset for details.
         *
         * @param preserveData determines if data on internal buffers should not be erased
         */
        void reset(bool preserveData = true);

        /**
         * Asynchronously copies data from internal host to internal device buffer.
         *
         */
        HINLINE void hostToDevice();

        /**
         * Asynchronously copies data from internal device to internal host buffer.
         */
        HINLINE void deviceToHost();

    private:
        std::unique_ptr<HBuffer> hostBuffer;
        std::unique_ptr<DBuffer> deviceBuffer;
    };

} // namespace pmacc

#include "pmacc/memory/buffers/HostDeviceBuffer.tpp"
