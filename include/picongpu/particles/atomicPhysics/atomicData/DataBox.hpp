/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            namespace atomicData
            {
                /** common interfaces of all data storage classes
                 *
                 * @tparam T_DataBoxType dataBox type used for storage
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_ConfigNumberDataType dataType used for storage of configNumber of atomic states
                 */
                template<typename T_Number, typename T_Value>
                class DataBox
                {
                public:
                    template<typename T_DataType>
                    using T_DataBoxType = pmacc::DataBox<pmacc::PitchedBox<T_DataType, 1u>>;

                    using BoxNumber = T_DataBoxType<T_Number>;
                    using BoxValue = T_DataBoxType<T_Value>;

                    using TypeNumber = T_Number;
                    using TypeValue = T_Value;
                };

            } // namespace atomicData
        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
