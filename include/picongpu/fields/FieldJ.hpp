/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/Fields.def"
#include "picongpu/particles/Particles.hpp"

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/types.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace picongpu
{
    /** Representation of the current density field
     *
     * Stores field values on host and device and provides data synchronization
     * between them.
     *
     * Implements interfaces defined by SimulationFieldHelper< MappingDesc > and
     * ISimulationData.
     */
    class FieldJ
        : public SimulationFieldHelper<MappingDesc>
        , public ISimulationData
    {
    public:
        //! Type of each field value
        using ValueType = float3_X;

        //! Number of components of ValueType, for serialization
        static constexpr int numComponents = ValueType::dim;

        //! Unit type of field components
        using UnitValueType = promoteType<float_64, ValueType>::type;

        //! Type of host-device buffer for field values
        using Buffer = pmacc::GridBuffer<ValueType, simDim>;

        //! Type of data box for field values on host and device
        using DataBoxType = DataBox<PitchedBox<ValueType, simDim>>;

        /** Create a field
         *
         * @param cellDescription mapping for kernels
         */
        FieldJ(MappingDesc const& cellDescription);

        //! Destroy a field
        virtual ~FieldJ() = default;

        //! Get a reference to the host-device buffer for the field values
        Buffer& getGridBuffer();

        //! Get the grid layout
        GridLayout<simDim> getGridLayout();

        //! Get the host data box for the field values
        DataBoxType getHostDataBox()
        {
            return buffer.getHostBuffer().getDataBox();
        }

        //! Get the device data box for the field values
        DataBoxType getDeviceDataBox()
        {
            return buffer.getDeviceBuffer().getDataBox();
        }

        /** Start asynchronous communication of field values
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
        void syncToDevice() override
        {
            ValueType tmp = float3_X(0., 0., 0.);
            buffer.getDeviceBuffer().setValue(tmp);
        }

        //! Synchronize host data with device data
        void synchronize() override;

        //! Get id
        SimulationDataId getUniqueId() override;

        //! Get units of field components
        HDINLINE static UnitValueType getUnit();

        /** Get unit representation as powers of the 7 base measures
         *
         * Characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         *  thermodynamic temperature theta, amount of substance N,
         *  luminous intensity J)
         */
        static std::vector<float_64> getUnitDimension();

        //! Get text name
        static std::string getName();

        /** Assign the given value to elements
         *
         * @param value value to assign all elements to
         */
        void assign(ValueType value);

        /** Compute current density created by a species in an area
         *
         * @tparam T_area area to compute currents in
         * @tparam T_Species particle species type
         *
         * @param species particle species
         * @param currentStep index of time iteration
         */
        template<uint32_t T_area, class T_Species>
        void computeCurrent(T_Species& species, uint32_t currentStep);

        /** Bash field in a direction.
         *
         * Copy all particles from the guard of a direction to the device exchange buffer
         *
         * @param exchangeType exchange type
         */
        void bashField(uint32_t exchangeType);

        /** Insert all fields which are in device exchange buffer
         *
         * @param exchangeType exchange type
         */
        void insertField(uint32_t exchangeType);

    private:
        //! Host-device buffer for current density values
        Buffer buffer;

        //! Buffer for receiving near-boundary values
        std::unique_ptr<Buffer> fieldJrecv;
    };

} // namespace picongpu
