/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */



#pragma once

#include <string>
#include <vector>

/*pic default*/
#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/Fields.def"
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/dataManagement/ISimulationData.hpp>

/*PMacc*/
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>

#include <pmacc/math/Vector.hpp>


namespace picongpu
{
    using namespace pmacc;

    class FieldB : public SimulationFieldHelper<MappingDesc>, public ISimulationData
    {
    public:
        typedef float3_X ValueType;
        typedef promoteType<float_64, ValueType>::type UnitValueType;
        static constexpr int numComponents = ValueType::dim;

        typedef DataBox<PitchedBox<ValueType, simDim> > DataBoxType;

        typedef MappingDesc::SuperCellSize SuperCellSize;

        FieldB( MappingDesc cellDescription);

        virtual ~FieldB();

        virtual void reset(uint32_t currentStep);

        HDINLINE static UnitValueType getUnit();

        /** powers of the 7 base measures
         *
         * characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         *  thermodynamic temperature theta, amount of substance N,
         *  luminous intensity J) */
        HINLINE static std::vector<float_64> getUnitDimension();

        static std::string getName();

        static uint32_t getCommTag();

        virtual EventTask asyncCommunication(EventTask serialEvent);

        DataBoxType getHostDataBox();

        GridLayout<simDim> getGridLayout();

        DataBoxType getDeviceDataBox();

        GridBuffer<ValueType, simDim> &getGridBuffer();

        SimulationDataId getUniqueId();

        void synchronize();

        void syncToDevice();

    private:

        void absorbeBorder();

        GridBuffer<ValueType, simDim> *fieldB;
    };


}
