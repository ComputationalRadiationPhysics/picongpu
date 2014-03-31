/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include "types.h"
#include "simulation_classTypes.hpp"

#include "fields/Fields.def"
#include "particles/ParticlesBase.hpp"
#include "particles/memory/buffers/ParticlesBuffer.hpp"

#include "dataManagement/ISimulationData.hpp"

#include <curand_kernel.h>

namespace picongpu
{
using namespace PMacc;

template<typename T_ParticleDescription>
class Particles : public ParticlesBase<T_ParticleDescription, MappingDesc>, public ISimulationData
{
public:

    typedef ParticlesBase<T_ParticleDescription, MappingDesc> ParticlesBaseType;
    typedef typename ParticlesBaseType::BufferType BufferType;
    typedef typename ParticlesBaseType::FrameType FrameType;
    typedef typename ParticlesBaseType::FrameTypeBorder FrameTypeBorder;
    typedef typename ParticlesBaseType::ParticlesBoxType ParticlesBoxType;


    Particles(GridLayout<simDim> gridLayout, MappingDesc cellDescription, SimulationDataId datasetID);

    virtual ~Particles();

    void createParticleBuffer(size_t gpuMemory);


    virtual void reset(uint32_t currentStep);

    void init(FieldE &fieldE, FieldB &fieldB, FieldJ &fieldJ, FieldTmp &fieldTmp);

    void update(uint32_t currentStep);

    void initFill(uint32_t currentStep);

    template< typename t_ParticleDescription>
    void deviceCloneFrom(Particles<t_ParticleDescription> &src);

    void deviceAddTemperature(float_X temperature);

    void deviceSetDrift(uint32_t currentStep);

    SimulationDataId getUniqueId();
    
    void synchronize();

    void syncToDevice();

private:
    SimulationDataId datasetID;
    GridLayout<simDim> gridLayout;


    FieldE *fieldE;
    FieldB *fieldB;
    FieldJ *fieldJurrent;
    FieldTmp *fieldTmp;

    curandState* randState;
};


} //namespace picongpu
