/* Copyright 2016-2021 Alexander Grund
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

#include "HostDeviceBuffer.hpp"


namespace pmacc
{
    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(const DataSpace<T_dim>& size, bool sizeOnDevice)
    {
        hostBuffer = new HostBufferIntern<T_Type, T_dim>(size);
        deviceBuffer = new DeviceBufferIntern<T_Type, T_dim>(size, sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(
        DBuffer& otherDeviceBuffer,
        const DataSpace<T_dim>& size,
        bool sizeOnDevice)
    {
        hostBuffer = new HostBufferIntern<T_Type, T_dim>(size);
        deviceBuffer = new DeviceBufferType(otherDeviceBuffer, size, DataSpace<T_dim>(), sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(
        HBuffer& otherHostBuffer,
        const DataSpace<T_dim>& offsetHost,
        DBuffer& otherDeviceBuffer,
        const DataSpace<T_dim>& offsetDevice,
        const GridLayout<T_dim> size,
        bool sizeOnDevice)
    {
        hostBuffer = new HostBufferType(otherHostBuffer, size, offsetHost);
        deviceBuffer = new DeviceBufferType(otherDeviceBuffer, size, offsetDevice, sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::~HostDeviceBuffer()
    {
        __delete(hostBuffer);
        __delete(deviceBuffer);
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
