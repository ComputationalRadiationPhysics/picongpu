/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera
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
 


#ifndef FIELDB_HPP
#define	FIELDB_HPP

#include <string>

/*pic default*/
#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_classTypes.hpp"


#include "fields/SimulationFieldHelper.hpp"
#include "dataManagement/ISimulationData.hpp"

/*libPMacc*/
#include "memory/buffers/GridBuffer.hpp"
#include "mappings/simulation/GridController.hpp"
#include "fields/LaserPhysics.hpp"
#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"

#include "basisLib/vector/Vector.hpp"


namespace picongpu
{
    using namespace PMacc;

    class FieldE;


    class FieldB : public SimulationFieldHelper<MappingDesc>, public ISimulationData
    {
    public:
        typedef float3_X FloatB;
        typedef typename promoteType<float_64, FloatB>::ValueType UnitValueType;
        static const int numComponents = FloatB::dim;
        
        typedef DataBox<PitchedBox<FloatB, simDim> > DataBoxType;

        static const uint32_t FloatBDim = simDim;
        typedef MappingDesc::SuperCellSize SuperCellSize;

        FieldB( MappingDesc cellDescription);

        virtual ~FieldB();

        virtual void reset(uint32_t currentStep);
        
        static UnitValueType getUnit();
        
        static std::string getName();
        
        static uint32_t getCommTag();

        virtual EventTask asyncCommunication(EventTask serialEvent);

        void init(FieldE &fieldE, LaserPhysics &laserPhysics);

        DataBoxType getHostDataBox();

        GridLayout<simDim> getGridLayout();

        DataBoxType getDeviceDataBox();

        GridBuffer<FloatB, simDim> &getGridBuffer();


        void synchronize();

        void syncToDevice();

    private:

        void absorbeBorder();

        void laserManipulation(uint32_t currentStep);

        GridBuffer<FloatB, simDim> *fieldB;

        FieldE *fieldE;
        LaserPhysics *laser;
    };


}

#include "fields/FieldB.tpp"

#endif	/* FIELDB_HPP */

