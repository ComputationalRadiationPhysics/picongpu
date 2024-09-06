/* Copyright 2013-2023 Axel Huebl, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Pawel Ordyna
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

#include "picongpu/defines.hpp"
#include "picongpu/fields/Fields.def"

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace picongpu
{
    /** Representation of the temporary scalar field for plugins and temporary
     *  particle data mapped to grid (charge density, energy density, etc.)
     *
     * Stores field values on host and device and provides data synchronization
     * between them.
     *
     * Implements interfaces defined by SimulationFieldHelper< MappingDesc > and
     * ISimulationData.
     */
    class FieldTmp
        : public SimulationFieldHelper<MappingDesc>
        , public ISimulationData
    {
    public:
        //! Type of each field value
        using ValueType = float1_X;

        //! Unit type of field components
        using UnitValueType = promoteType<float_64, ValueType>::type;

        //! Size of supercell
        using SuperCellSize = MappingDesc::SuperCellSize;

        //! Type of data box for field values on host and device
        using DataBoxType = DataBox<PitchedBox<ValueType, simDim>>;

        /** Create a field
         *
         * @param cellDescription mapping for kernels
         * @param slotId index of the temporary field
         */
        FieldTmp(MappingDesc const& cellDescription, uint32_t slotId);

        //! Destroy a field
        ~FieldTmp() override = default;

        //! Get a reference to the host-device buffer for the field values
        GridBuffer<ValueType, simDim>& getGridBuffer();

        //! Get the grid layout
        GridLayout<simDim> getGridLayout();

        //! Get the host data box for the field values
        DataBoxType getHostDataBox();

        //! Get the device data box for the field values
        DataBoxType getDeviceDataBox();

        /** Start asynchronous send of field values
         *
         * Add data from the local guard of the GPU to the border of the neighboring GPUs.
         * This method can be called before or after asyncCommunicationGather without
         * explicit handling to avoid race conditions between both methods.
         *
         * @param serialEvent event to depend on
         */
        virtual EventTask asyncCommunication(EventTask serialEvent);

        /** Reset the host-device buffer for field values
         *
         * @param currentStep index of time iteration
         */
        void reset(uint32_t currentStep) override;

        //! Synchronize device data with host data
        void syncToDevice() override;

        //! Synchronize host data with device data
        void synchronize() override;

        /** Get id
         *
         * @param slotId index of the temporary field
         */
        static SimulationDataId getUniqueId(uint32_t slotId);

        //! Get id
        SimulationDataId getUniqueId() override;

        //! Get unit of field components
        template<class FrameSolver>
        static UnitValueType getUnit()
        {
            return FrameSolver().getUnit();
        }

        /** Get unit representation as powers of the 7 base measures
         *
         * Characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         *  thermodynamic temperature theta, amount of substance N,
         *  luminous intensity J)
         */
        template<class FrameSolver>
        static std::vector<float_64> getUnitDimension()
        {
            return FrameSolver().getUnitDimension();
        }

        //! Get mapping for kernels
        MappingDesc getCellDescription()
        {
            return this->cellDescription;
        }

        //! Get text name
        static std::string getName();

        /** Gather data from neighboring GPUs
         *
         * Copy data from the border of neighboring GPUs into the local guard.
         * This method can be called before or after asyncCommunication without
         * explicit handling to avoid race conditions between both methods.
         */
        EventTask asyncCommunicationGather(EventTask serialEvent);

        /** Bash particles in a direction.
         * Copy all particles from the guard of a direction to the device exchange buffer
         *
         * @param exchangeType exchange type
         */
        void bashField(uint32_t exchangeType);

        /** Insert all particles which are in device exchange buffer
         *
         * @param exchangeType exchange type
         */
        void insertField(uint32_t exchangeType);

    private:
        //! Host-device buffer for current density values
        std::unique_ptr<GridBuffer<ValueType, simDim>> fieldTmp;

        //! Buffer for receiving near-boundary values
        std::unique_ptr<GridBuffer<ValueType, simDim>> fieldTmpRecv;

        //! Index of the temporary field
        uint32_t m_slotId;

        //! Events for communication
        EventTask m_scatterEv;
        EventTask m_gatherEv;

        //! Tags for communication
        uint32_t m_commTagScatter;
        uint32_t m_commTagGather;
    };

} // namespace picongpu
