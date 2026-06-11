/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "HostDeviceBuffer.hpp"

namespace pmacc
{
    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(DataSpace<T_dim> const& size, bool sizeOnDevice)
    {
        hostBuffer = std::make_unique<HostBuffer<T_Type, T_dim>>(size);
        deviceBuffer = std::make_unique<DeviceBuffer<T_Type, T_dim>>(size, sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(
        DBuffer& otherDeviceBuffer,
        DataSpace<T_dim> const& size,
        bool sizeOnDevice)
    {
        hostBuffer = std::make_unique<HostBuffer<T_Type, T_dim>>(size);
        deviceBuffer = std::make_unique<DeviceBufferType>(otherDeviceBuffer, size, DataSpace<T_dim>(), sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(
        HBuffer& otherHostBuffer,
        DataSpace<T_dim> const& offsetHost,
        DBuffer& otherDeviceBuffer,
        DataSpace<T_dim> const& offsetDevice,
        GridLayout<T_dim> const size,
        bool sizeOnDevice)
    {
        hostBuffer = std::make_unique<HostBufferType>(otherHostBuffer, size, offsetHost);
        deviceBuffer = std::make_unique<DeviceBufferType>(otherDeviceBuffer, size, offsetDevice, sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostBuffer<T_Type, T_dim>& HostDeviceBuffer<T_Type, T_dim>::getHostBuffer() const
    {
        return *hostBuffer;
    }

    template<typename T_Type, unsigned T_dim>
    DeviceBuffer<T_Type, T_dim>& HostDeviceBuffer<T_Type, T_dim>::getDeviceBuffer() const
    {
        return *deviceBuffer;
    }

    template<typename T_Type, unsigned T_dim>
    void HostDeviceBuffer<T_Type, T_dim>::reset(bool preserveData)
    {
        deviceBuffer->reset(preserveData);
        hostBuffer->reset(preserveData);
    }

    template<typename T_Type, unsigned T_dim>
    void HostDeviceBuffer<T_Type, T_dim>::hostToDevice()
    {
        deviceBuffer->copyFrom(*hostBuffer);
    }

    template<typename T_Type, unsigned T_dim>
    void HostDeviceBuffer<T_Type, T_dim>::deviceToHost()
    {
        hostBuffer->copyFrom(*deviceBuffer);
    }

} // namespace pmacc
