/**
 * Copyright 2013-2016 Felix Schmitt, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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

#include "particles/ParticlesBase.kernel"
#include "fields/SimulationFieldHelper.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "particles/memory/boxes/ParticlesBox.hpp"
#include "particles/memory/buffers/ParticlesBuffer.hpp"

#include "mappings/kernel/StrideMapping.hpp"
#include "traits/NumberOfExchanges.hpp"
#include "assert.hpp"


namespace PMacc
{

/* Tag used for marking particle types */
struct ParticlesTag;

template<typename T_ParticleDescription, class T_MappingDesc>
class ParticlesBase : public SimulationFieldHelper<T_MappingDesc>
{
    typedef T_ParticleDescription ParticleDescription;
    typedef T_MappingDesc MappingDesc;

public:

    /* Type of used particles buffer
     */
    typedef ParticlesBuffer<ParticleDescription, typename MappingDesc::SuperCellSize, MappingDesc::Dim> BufferType;

    /* Type of frame in particles buffer
     */
    typedef typename BufferType::FrameType FrameType;
    /* Type of border frame in a particle buffer
     */
    typedef typename BufferType::FrameTypeBorder FrameTypeBorder;

    /* Type of the particle box which particle buffer create
     */
    typedef ParticlesBox< FrameType, MappingDesc::Dim> ParticlesBoxType;

    /* Policies for handling particles in guard cells */
    typedef typename ParticleDescription::HandleGuardRegion HandleGuardRegion;

    enum
    {
        Dim = MappingDesc::Dim,
        Exchanges = traits::NumberOfExchanges<Dim>::value,
        TileSize = math::CT::volume<typename MappingDesc::SuperCellSize>::type::value
    };

    /* Mark this simulation data as a particle type */
    typedef ParticlesTag SimulationDataTag;

protected:

    BufferType *particlesBuffer;

    ParticlesBase(MappingDesc description) : SimulationFieldHelper<MappingDesc>(description), particlesBuffer(NULL)
    {
    }

    virtual ~ParticlesBase(){}

    /* Shift all particle in a AREA
     * @tparam AREA area which is used (CORE,BORDER,GUARD or a combination)
     */
    template<uint32_t AREA>
    void shiftParticles()
    {
        StrideMapping<AREA, 3, MappingDesc> mapper(this->cellDescription);
        ParticlesBoxType pBox = particlesBuffer->getDeviceParticleBox();

        __startTransaction(__getTransactionEvent());
        do
        {
            PMACC_KERNEL(KernelShiftParticles{})
                (mapper.getGridDim(), (int)TileSize)
                (pBox, mapper);
            PMACC_KERNEL(KernelFillGaps{})
                (mapper.getGridDim(), (int)TileSize)
                (pBox, mapper);
            PMACC_KERNEL(KernelFillGapsLastFrame{})
                (mapper.getGridDim(), (int)TileSize)
                (pBox, mapper);
        }
        while (mapper.next());

        __setTransactionEvent(__endTransaction());

    }

    /* fill gaps in a AREA
     * @tparam AREA area which is used (CORE,BORDER,GUARD or a combination)
     */
    template<uint32_t AREA>
    void fillGaps()
    {
        AreaMapping<AREA, MappingDesc> mapper(this->cellDescription);

        PMACC_KERNEL(KernelFillGaps{})
            (mapper.getGridDim(), (int)TileSize)
            (particlesBuffer->getDeviceParticleBox(), mapper);

        PMACC_KERNEL(KernelFillGapsLastFrame{})
            (mapper.getGridDim(), (int)TileSize)
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

    /* Delete all particles in GUARD for one direction.
     */
    void deleteGuardParticles(uint32_t exchangeType);

    /* Delete all particle in an area*/
    template<uint32_t T_area>
    void deleteParticlesInArea();

    /* Bash particles in a direction.
     * Copy all particles from the guard of a direction to the device exchange buffer
     */
    void bashParticles(uint32_t exchangeType);

    /* Insert all particles which are in device exchange buffer
     */
    void insertParticles(uint32_t exchangeType);

    ParticlesBoxType getDeviceParticlesBox()
    {
        return particlesBuffer->getDeviceParticleBox();
    }

    ParticlesBoxType getHostParticlesBox(const int64_t memoryOffset)
    {
        return particlesBuffer->getHostParticleBox(memoryOffset);
    }

    /* Get the particles buffer which is used for the particles.
     */
    BufferType& getParticlesBuffer()
    {
        PMACC_ASSERT(particlesBuffer != NULL);
        return *particlesBuffer;
    }

    /* set all internal objects to initial state*/
    virtual void reset(uint32_t currentStep);

};

} //namespace PMacc

#include "particles/ParticlesBase.tpp"
