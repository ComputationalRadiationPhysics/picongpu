/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
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
#include "pmacc_types.hpp"
#include "simulation_defines.hpp"
#include "simulation_classTypes.hpp"

#include "Fields.def"
#include "fields/SimulationFieldHelper.hpp"
#include "dataManagement/ISimulationData.hpp"

/*libPMacc*/
#include "memory/buffers/GridBuffer.hpp"
#include "mappings/simulation/GridController.hpp"
#include "fields/LaserPhysics.def"
#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"

#include "math/Vector.hpp"


namespace picongpu
{
    using namespace PMacc;

    class FieldE: public SimulationFieldHelper<MappingDesc>, public ISimulationData
    {
    public:
        typedef float3_X ValueType;
        typedef promoteType<float_64, ValueType>::type UnitValueType;
        static constexpr int numComponents = ValueType::dim;

        typedef MappingDesc::SuperCellSize SuperCellSize;

        typedef DataBox<PitchedBox<ValueType, simDim> > DataBoxType;


        FieldE(MappingDesc cellDescription);

        virtual ~FieldE();

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

        void init(FieldB &fieldB,LaserPhysics &laserPhysics);

        DataBoxType getDeviceDataBox();

        DataBoxType getHostDataBox();

        GridBuffer<ValueType,simDim>& getGridBuffer();

        GridLayout<simDim> getGridLayout();

        SimulationDataId getUniqueId();

        void synchronize();

        void syncToDevice();

        void laserManipulation(uint32_t currentStep);

    private:

        void absorbeBorder();


        GridBuffer<ValueType,simDim> *fieldE;

        FieldB *fieldB;

        LaserPhysics *laser;
    };


} // namespace picongpu
