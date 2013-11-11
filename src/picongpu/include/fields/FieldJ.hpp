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
 


#ifndef FIELDJ_HPP
#define	FIELDJ_HPP

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

#include "math/Vector.hpp"

namespace picongpu
{
    using namespace PMacc;

    template< typename,typename>
    class Particles;
    
    // The fieldJ saves the current density j
    //
    // j = current / area
    // To obtain the current which goes out of a cell in the 3 directions,
    // calculate J = float3_X( j.x() * cellSize.y() * cellSize.z(),
    //                            j.y() * cellSize.x() * cellSize.z(),
    //                            j.z() * cellSize.x() * cellSize.y())
    //
    class FieldJ: public SimulationFieldHelper<MappingDesc>, public ISimulationData
    {
    public:
        typedef float3_X FloatJ;
        typedef typename promoteType<float_64, FloatJ>::ValueType UnitValueType;
        static const int numComponents = FloatJ::dim;
        
        static const uint32_t FloatJDim = simDim;
        
        typedef DataBox<PitchedBox<FloatJ, simDim> > DataBoxType;
                
        FieldJ(MappingDesc cellDescription);

        virtual ~FieldJ();

        virtual EventTask asyncCommunication(EventTask serialEvent);

        void init(FieldE &fieldE);

        GridLayout<simDim> getGridLayout();

        void reset(uint32_t currentStep);
        
        void clear();
        
        static UnitValueType getUnit();
        
        static std::string getName();
        
        static uint32_t getCommTag();
        
        template<uint32_t AREA,class ParticlesClass>
        void computeCurrent(ParticlesClass &parClass,uint32_t currentStep) throw(std::invalid_argument);
        
        template<uint32_t AREA>
        void addCurrentToE();

        void synchronize();
        
        void syncToDevice()
        {
            FloatJ tmp=float3_X(0.,0.,0.);
            fieldJ.getDeviceBuffer().setValue(tmp);
        }
        
        DataBoxType getDeviceDataBox() {return fieldJ.getDeviceBuffer().getDataBox();}
        
        DataBoxType getHostDataBox() {return fieldJ.getHostBuffer().getDataBox();}
        
        GridBuffer<FloatJ, simDim> &getGridBuffer();
        
        /* Bash particles in a direction.
         * Copy all particles from the guard of a direction to the device exchange buffer
         */
        void bashField(uint32_t exchangeType);

        /* Insert all particles which are in device exchange buffer
         */
        void insertField(uint32_t exchangeType);

    private:
        
        GridBuffer<FloatJ, FloatJDim> fieldJ;

        FieldE *fieldE;
    };


} //namespace

#include "fields/FieldJ.tpp"

#endif	/* FIELDJ_HPP */

