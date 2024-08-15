/* Copyright 2022-2023 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
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
