/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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
#include "fields/SimulationFieldHelper.hpp"

#include "particles/ParticlesBase.hpp"
#include "particles/memory/buffers/ParticlesBuffer.hpp"

#include "dataManagement/ISimulationData.hpp"

#include <curand_kernel.h>

namespace picongpu
{
using namespace PMacc;

class FieldJ;
class FieldB;
class FieldE;

template< typename T_DataVector, typename T_MethodsVector>
class Particles : public ParticlesBase<T_DataVector,T_MethodsVector, MappingDesc>, public ISimulationData
{
public:

    typedef ParticlesBase<T_DataVector,T_MethodsVector, MappingDesc> ParticlesBaseType;
    typedef typename ParticlesBaseType::BufferType BufferType;
    typedef typename ParticlesBaseType::FrameType FrameType;
    typedef typename ParticlesBaseType::FrameTypeBorder FrameTypeBorder;
    typedef typename ParticlesBaseType::ParticlesBoxType ParticlesBoxType;


    Particles(GridLayout<simDim> gridLayout, MappingDesc cellDescription);

    virtual ~Particles();

    void createParticleBuffer(size_t gpuMemory);


    virtual void reset(uint32_t currentStep);

    void init(FieldE &fieldE, FieldB &fieldB, FieldJ &fieldJ, int datasetID);

    void update(uint32_t currentStep);

    void initFill(uint32_t currentStep);

    template< typename t_DataVector,typename t_MethodsVector>
    void deviceCloneFrom(Particles<t_DataVector,t_MethodsVector> &src);

    void deviceAddTemperature(float_X temperature);

    void deviceSetDrift(uint32_t currentStep);

    void synchronize();

    void syncToDevice();

private:
    int datasetID;
    GridLayout<simDim> gridLayout;


    FieldE *fieldE;
    FieldB *fieldB;
    FieldJ *fieldJurrent;


    curandState* randState;
};


} //namespace picongpu

#include "particles/Particles.tpp"
