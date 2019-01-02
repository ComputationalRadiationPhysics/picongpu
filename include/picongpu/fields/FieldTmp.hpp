/* Copyright 2013-2019 Axel Huebl, Rene Widera, Richard Pausch,
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

#include <memory>


namespace picongpu
{
    using namespace pmacc;


    /** Tmp (at the moment: scalar) field for plugins and tmp data like
     *  "gridded" particle data (charge density, energy density, ...)
     */
    class FieldTmp :
        public SimulationFieldHelper<MappingDesc>,
        public ISimulationData
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

        FieldTmp(
            MappingDesc cellDescription,
            uint32_t slotId
        );

        virtual ~FieldTmp( );

        virtual void reset( uint32_t currentStep );

        template< class FrameSolver >
        HDINLINE static UnitValueType getUnit();

        /** powers of the 7 base measures
         *
         * characterizing the record's unit in SI
         * (length L, mass M, time T, electric current I,
         *  thermodynamic temperature theta, amount of substance N,
         *  luminous intensity J) */
        template<class FrameSolver >
        HINLINE static std::vector<float_64> getUnitDimension();

        static std::string getName();

        /** scatter data to neighboring GPUs
         *
         * Add data from the local guard of the GPU to the border of the neighboring GPUs.
         * This method can be called before or after asyncCommunicationGather without
         * explicit handling to avoid race conditions between both methods.
         */
        virtual EventTask asyncCommunication( EventTask serialEvent );

        /** gather data from neighboring GPUs
         *
         * Copy data from the border of neighboring GPUs into the local guard.
         * This method can be called before or after asyncCommunication without
         * explicit handling to avoid race conditions between both methods.
         */
        EventTask asyncCommunicationGather( EventTask serialEvent );

        DataBoxType getDeviceDataBox( );

        DataBoxType getHostDataBox( );

        GridBuffer<ValueType, simDim>& getGridBuffer( );

        GridLayout<simDim> getGridLayout( );

        template<uint32_t AREA, class FrameSolver, class ParticlesClass>
        void computeValue(ParticlesClass& parClass, uint32_t currentStep);

        static SimulationDataId getUniqueId( uint32_t slotId );

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

        std::unique_ptr< GridBuffer<ValueType, simDim> > fieldTmp;
        std::unique_ptr< GridBuffer<ValueType, simDim> > fieldTmpRecv;

        uint32_t m_slotId;

        EventTask m_scatterEv;
        uint32_t m_commTagScatter;
        EventTask m_gatherEv;
        uint32_t m_commTagGather;
    };


}
