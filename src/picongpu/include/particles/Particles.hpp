/**
 * Copyright 2013-2015 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Marco Garten, Alexander Grund
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
#include "fields/Fields.hpp"
#include "particles/ParticlesBase.hpp"
#include "particles/memory/buffers/ParticlesBuffer.hpp"
#include "particles/manipulators/manipulators.def"

#include "dataManagement/ISimulationData.hpp"

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

    void createParticleBuffer();

    void init(FieldE &fieldE, FieldB &fieldB, FieldJ &fieldJ, FieldTmp &fieldTmp);

    void update(uint32_t currentStep);

    template<typename T_GasFunctor, typename T_PositionFunctor>
    void initGas(T_GasFunctor& gasFunctor, T_PositionFunctor& positionFunctor, const uint32_t currentStep);

    template< typename T_SrcParticleDescription,
              typename T_ManipulateFunctor>
    void deviceCloneFrom(Particles<T_SrcParticleDescription> &src, T_ManipulateFunctor& manipulateFunctor);

    template<typename T_Functor>
    void manipulateAllParticles(uint32_t currentStep, T_Functor& functor);

    SimulationDataId getUniqueId();

    /* sync device data to host
     *
     * ATTENTION: - in the current implementation only supercell meta data are copied!
     *            - the shared (between all species) mallocMC buffer must be copied once
     *              by the user
     */
    void synchronize();

    void syncToDevice();

private:
    SimulationDataId datasetID;
    GridLayout<simDim> gridLayout;


    FieldE *fieldE;
    FieldB *fieldB;
    FieldJ *fieldJcurrent;
    FieldTmp *fieldTmp;
};

namespace traits
{
template<typename T_ParticleDescription>
struct GetDataBoxType<picongpu::Particles<T_ParticleDescription> >
{
    typedef typename picongpu::Particles<T_ParticleDescription>::ParticlesBoxType type;
};

} //namespace traits
} //namespace picongpu
