/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/Fields.def"
#include "picongpu/simulation_types.hpp"

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>


namespace picongpu
{
    namespace fields
    {
        /** Base class for implementation inheritance in classes for the
         *  electromagnetic fields
         *
         * Stores field values on host and device and provides data synchronization
         * between them.
         *
         * Implements interfaces defined by SimulationFieldHelper< MappingDesc > and
         * ISimulationData.
         */
        class EMFieldBase
            : public SimulationFieldHelper<MappingDesc>
            , public ISimulationData
        {
        public:
            //! Type of each field value
            using ValueType = float3_X;

            //! Number of components of ValueType, for serialization
            static constexpr int numComponents = ValueType::dim;

            //! Type of host-device buffer for field values
            using Buffer = pmacc::GridBuffer<ValueType, simDim>;

            //! Type of data box for field values on host and device
            using DataBoxType = pmacc::DataBox<PitchedBox<ValueType, simDim>>;

            //! Size of supercell
            using SuperCellSize = MappingDesc::SuperCellSize;

            /** Create a field
             *
             * @tparam T_DerivedField derived field type, needed for compile-time information
             *
             * @param cellDescription mapping for kernels
             * @param id unique id
             * @param placeholder parameter used for type deduction in this template constructor,
             *                    is not used otherwise
             */
            template<typename T_DerivedField>
            HINLINE EMFieldBase(
                MappingDesc const& cellDescription,
                pmacc::SimulationDataId const& id,
                T_DerivedField const& placeholder);

            //! Get a reference to the host-device buffer for the field values
            HINLINE Buffer& getGridBuffer();

            //! Get the grid layout
            HINLINE GridLayout<simDim> getGridLayout();

            //! Get the host data box for the field values
            HINLINE DataBoxType getHostDataBox();

            //! Get the device data box for the field values
            HINLINE DataBoxType getDeviceDataBox();

            /** Start asynchronous communication of field values
             *
             * @param serialEvent event to depend on
             */
            HINLINE EventTask asyncCommunication(EventTask serialEvent);

            /** Reset the host-device buffer for field values
             *
             * @param currentStep index of time iteration
             */
            HINLINE void reset(uint32_t currentStep) override;

            //! Synchronize device data with host data
            HINLINE void syncToDevice() override;

            //! Synchronize host data with device data
            HINLINE void synchronize() override;

            //! Get id
            HINLINE SimulationDataId getUniqueId() override;

        private:
            //! Host-device buffer for field values
            std::unique_ptr<Buffer> buffer;

            //! Unique id
            pmacc::SimulationDataId id;
        };

    } // namespace fields
} // namespace picongpu
