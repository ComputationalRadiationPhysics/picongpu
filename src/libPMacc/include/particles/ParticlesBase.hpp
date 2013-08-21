/**
 * Copyright 2013 Felix Schmitt, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 
#pragma once

#include "eventSystem/EventSystem.hpp"

#include "particles/ParticlesBase.kernel"
#include "fields/SimulationFieldHelper.hpp"
#include "mappings/kernel/ExchangeMapping.hpp"

#include "particles/tasks/ParticleFactory.hpp"

#include "particles/memory/boxes/ParticlesBox.hpp"
#include "particles/memory/buffers/ParticlesBuffer.hpp"

#include "mappings/kernel/StrideMapping.hpp"


namespace PMacc
{

template<typename PositionType, class UserTypeList, class MappingDesc>
class ParticlesBase : public SimulationFieldHelper<MappingDesc>
{
public:

    /* Type of used partciles buffer
     */
    typedef ParticlesBuffer<PositionType, UserTypeList, typename MappingDesc::SuperCellSize, MappingDesc::Dim> BufferType;

    /* Type of frame in particles buffer
     */
    typedef typename BufferType::ParticleType FrameType;
    /* Type of boder frame in a particle buffer
     */
    typedef typename BufferType::ParticleTypeBorder FrameTypeBorder;

    /* TYpe of the particle box which particle buffer create
     */
    typedef ParticlesBox< FrameType, MappingDesc::Dim> ParticlesBoxType;

protected:

    enum
    {
        Dim = MappingDesc::Dim,
        /* Exchanges in 2D=9 and in 3D=27
         */
        Exchanges = MappingDesc::Neighbors + 1,
        TileSize = MappingDesc::SuperCellSize::elements
    };

    BufferType *particlesBuffer;

    ParticlesBase(MappingDesc description) : SimulationFieldHelper<MappingDesc>(description), particlesBuffer(NULL)
    {
    }

    /* Shift all particle in a AREA
     * @tparam AREA area whish is used (CORE,BORDER,GUARD or a combination)
     */
    template<uint32_t AREA>
    void shiftParticles()
    {
        StrideMapping<AREA, DIM3, MappingDesc> mapper(this->cellDescription);
        ParticlesBoxType pBox = particlesBuffer->getDeviceParticleBox();

        __startTransaction(__getTransactionEvent());
        do
        {
            __cudaKernel(kernelShiftParticles)
                (mapper.getGridDim(), TileSize)
                (pBox, mapper);
        }
        while (mapper.next());

        __setTransactionEvent(__endTransaction());

    }

    /* fill gaps in a AREA
     * @tparam AREA area whish is used (CORE,BORDER,GUARD or a combination)
     */
    template<uint32_t AREA>
    void fillGaps()
    {
        AreaMapping<AREA, MappingDesc> mapper(this->cellDescription);

        __cudaKernel(kernelFillGaps)
            (mapper.getGridDim(), TileSize)
            (particlesBuffer->getDeviceParticleBox(), mapper);

        __cudaKernel(kernelFillGapsLastFrame)
            (mapper.getGridDim(), TileSize)
            (particlesBuffer->getDeviceParticleBox(), mapper);
    }


public:

    /* fill gaps in a the complete simulation area (include GUARD)
     */
    void fillAllGaps()
    {
        this->fillGaps < CORE + BORDER + GUARD > ();
    }

    /* fill all gaps in the border of the simulation
     */
    void fillBorderGaps()
    {
        this->fillGaps < BORDER > ();
    }

    /* Bash particles in a direction.
     * Copy all particles from the guard of a direction to the device exchange buffer
     */
    void bashParticles(uint32_t exchangeType);

    /* Insert all particles which are in device exchange buffer
     */
    void insertParticles(uint32_t exchangeType);

    ParticlesBoxType getHostParticlesBox()
    {
        return particlesBuffer->getHostParticleBox();
    }

    ParticlesBoxType getDeviceParticlesBox()
    {
        return particlesBuffer->getDeviceParticleBox();
    }

    /* Get the particles buffer which is used for the particles.
     */
    BufferType& getParticlesBuffer()
    {
        assert(particlesBuffer != NULL);
        return *particlesBuffer;
    }

    /* Communicate particles to neighbor devices.
     * This method include bashing and insert of particles full
     * asynchron.
     */
    EventTask asyncCommunication(EventTask event);
};

} //namespace PMacc

#include "particles/ParticlesBase.tpp"


