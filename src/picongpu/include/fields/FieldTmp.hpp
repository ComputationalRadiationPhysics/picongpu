/**
 * Copyright 2013-2016 Axel Huebl, Rene Widera, Richard Pausch,
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

#include "fields/Fields.def"
#include "fields/SimulationFieldHelper.hpp"
#include "dataManagement/ISimulationData.hpp"

/*libPMacc*/
#include "memory/buffers/GridBuffer.hpp"
#include "mappings/simulation/GridController.hpp"
#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"


namespace picongpu
{
    using namespace PMacc;


    /** Tmp (at the moment: scalar) field for analysers and tmp data like
     *  "gridded" particle data (charge density, energy density, ...)
     */
    class FieldTmp : public SimulationFieldHelper<MappingDesc>, public ISimulationData
    {
    public:
        typedef float1_X ValueType;
        typedef promoteType<float_64, ValueType>::type UnitValueType;

        typedef MappingDesc::SuperCellSize SuperCellSize;
        typedef DataBox<PitchedBox<ValueType, simDim> > DataBoxType;

        MappingDesc getCellDescription()
        {
            return this->cellDescription;
        }

        FieldTmp( MappingDesc cellDescription );

        virtual ~FieldTmp( );

        virtual void reset( uint32_t currentStep );

        template<class FrameSolver >
        HDINLINE static UnitValueType getUnit();

        /** powers of the 7 base measures
         *
         * characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         *  thermodynamic temperature theta, amount of substance N,
         *  luminous intensity J) */
        template<class FrameSolver >
        HDINLINE static std::vector<float_64> getUnitDimension();

        static std::string getName();

        static uint32_t getCommTag();

        virtual EventTask asyncCommunication( EventTask serialEvent );

        void init( );

        DataBoxType getDeviceDataBox( );

        DataBoxType getHostDataBox( );

        GridBuffer<ValueType, simDim>& getGridBuffer( );

        GridLayout<simDim> getGridLayout( );

        template<uint32_t AREA, class FrameSolver, class ParticlesClass>
        void computeValue(ParticlesClass& parClass, uint32_t currentStep);

        SimulationDataId getUniqueId();

        void synchronize( );

        void syncToDevice( );

        /* Bash particles in a direction.
         * Copy all particles from the guard of a direction to the device exchange buffer
         */
        void bashField( uint32_t exchangeType );

        /* Insert all particles which are in device exchange buffer
         */
        void insertField( uint32_t exchangeType );

    private:
        GridBuffer<ValueType, simDim> *fieldTmp;

    };


}
