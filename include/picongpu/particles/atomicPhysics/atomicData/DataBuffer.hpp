/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/memory/buffers/HostDeviceBuffer.tpp>

namespace picongpu::particles::atomicPhysics::atomicData
{
    /** common interfaces of all buffer data storage classes
     *
     * @tparam T_DataBoxType dataBox type used for storage
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     */
    template<typename T_Number, typename T_Value>
    class DataBuffer
    {
    public:
        using BufferNumber = pmacc::HostDeviceBuffer<T_Number, 1u>;
        using BufferValue = pmacc::HostDeviceBuffer<T_Value, 1u>;

        using TypeNumber = T_Number;
        using TypeValue = T_Value;
    };
} // namespace picongpu::particles::atomicPhysics::atomicData
