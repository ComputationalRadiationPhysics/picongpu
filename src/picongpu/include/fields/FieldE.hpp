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
 


#ifndef FIELDE_HPP
#define	FIELDE_HPP

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

    class FieldB;

    class FieldE: public SimulationFieldHelper<MappingDesc>, public ISimulationData
    {
    public:
        typedef float3_X FloatE;
        typedef typename promoteType<float_64, FloatE>::ValueType UnitValueType;
        static const int numComponents = FloatE::dim;
        
        static const uint32_t FloatEDim = simDim;
        typedef MappingDesc::SuperCellSize SuperCellSize;
        
        typedef DataBox<PitchedBox<FloatE, simDim> > DataBoxType;


        FieldE(MappingDesc cellDescription);

        virtual ~FieldE();

        virtual void reset(uint32_t currentStep);
        
        static UnitValueType getUnit();
        
        static std::string getName();
        
        static uint32_t getCommTag();

        virtual EventTask asyncCommunication(EventTask serialEvent);

        void init(FieldB &fieldB,LaserPhysics &laserPhysics);

        DataBoxType getDeviceDataBox();

        DataBoxType getHostDataBox();
        
        GridBuffer<FloatE,simDim>& getGridBuffer();

        GridLayout<simDim> getGridLayout();

        void synchronize();
        
        void syncToDevice();
        
        void laserManipulation(uint32_t currentStep);

    private:

        void absorbeBorder();
        

        GridBuffer<FloatE,simDim> *fieldE;

        FieldB *fieldB;

        LaserPhysics *laser;
    };


}

#include "fields/FieldE.tpp"

#endif	/* FIELDE_HPP */

