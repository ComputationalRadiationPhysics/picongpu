/* Copyright 2022-2024 Brian Marre
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

/** @file implements a superCell based field */

#pragma once

#include "picongpu/simulation_defines.hpp"
// need simDim from picongpu/param/dimension.param

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics
{
    /** holds a gridBuffer of one entry per super cell
     *
     * @tparam T_Entry type of the entry to store for each superCell
     * @tparam T_MappingDescription type used for description of mapping of
     *      simulation domain to memory
     * @tparam T_withGuards true =^= create with guards, false =^= create without guards
     */
    template<typename T_Entry, typename T_MappingDescription, bool T_withGuards>
    struct SuperCellField
        : public pmacc::ISimulationData
        , public pmacc::SimulationFieldHelper<T_MappingDescription>
    {
        //! type of data box for field values on host and device
        using DataBoxType = pmacc::DataBox<pmacc::PitchedBox<T_Entry, picongpu::simDim>>;
        //! type of device buffer
        using DeviceBufferType = pmacc::DeviceBuffer<T_Entry, picongpu::simDim>;
        //! type of entry
        using entryType = T_Entry;

        constexpr static bool hasGuards = T_withGuards;

        /* from SimulationFieldHelper<T_MappingDescription>:
         * protected:
         *      T_MappingDescription cellDescription;
         * public:
         *      static constexpr uint32_t dim = T_MappingDescription::dim;
         *      using MappingDesc = T_MappingDescription;
         *      void ~<> = default;
         *
         *      T_MappingDescription getCellDescription() const
         */

        /// @todo should be private?, Brian Marre, 2022
        //! pointer to gridBuffer of histograms:T_histograms created upon creation
        std::unique_ptr<pmacc::GridBuffer<T_Entry, picongpu::simDim>> superCellField;

        SuperCellField(T_MappingDescription const& mappingDesc)
            : SimulationFieldHelper<T_MappingDescription>(mappingDesc)
        {
            if constexpr(T_withGuards)
                superCellField
                    = std::make_unique<GridBuffer<T_Entry, picongpu::simDim>>(mappingDesc.getGridSuperCells());
            else
                superCellField = std::make_unique<GridBuffer<T_Entry, picongpu::simDim>>(
                    mappingDesc.getGridSuperCellsWithoutGuards());
        }

        // required by ISimulationData
        virtual std::string getUniqueId() override = 0;

        // required by ISimulationData
        //! == deviceToHost
        HINLINE void synchronize() override
        {
            superCellField->deviceToHost();
        }

        // required by SimulationFieldHelper
        HINLINE void reset(uint32_t currentStep) override
        {
            superCellField->getHostBuffer().reset(true);
            superCellField->getDeviceBuffer().reset(false);
        };

        // required by SimulationHelperField
        //! ==hostToDevice
        HINLINE void syncToDevice() override
        {
            superCellField->hostToDevice();
        }

        /** get dataBox on device for use in device kernels
         *
         * Note: dataBoxes are just "pointers"
         */
        HINLINE DataBoxType getDeviceDataBox()
        {
            return superCellField->getDeviceBuffer().getDataBox();
        }

        HINLINE DeviceBufferType& getDeviceBuffer()
        {
            return superCellField->getDeviceBuffer();
        }

        HINLINE GridLayout<picongpu::simDim> getGridLayout()
        {
            return superCellField->getGridLayout();
        }
    };
} // namespace picongpu::particles::atomicPhysics
