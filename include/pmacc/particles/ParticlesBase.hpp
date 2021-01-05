/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/particles/ParticlesBase.kernel"
#include "pmacc/fields/SimulationFieldHelper.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"

#include "pmacc/particles/memory/boxes/ParticlesBox.hpp"
#include "pmacc/particles/memory/buffers/ParticlesBuffer.hpp"

#include "pmacc/mappings/kernel/StrideMapping.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"
#include "pmacc/assert.hpp"

#include <memory>


namespace pmacc
{
    /* Tag used for marking particle types */
    struct ParticlesTag;

    template<typename T_ParticleDescription, class T_MappingDesc, typename T_DeviceHeap>
    class ParticlesBase : public SimulationFieldHelper<T_MappingDesc>
    {
        typedef T_ParticleDescription ParticleDescription;
        typedef T_MappingDesc MappingDesc;

    public:
        /* Type of used particles buffer
         */
        typedef ParticlesBuffer<
            ParticleDescription,
            typename MappingDesc::SuperCellSize,
            T_DeviceHeap,
            MappingDesc::Dim>
            BufferType;

        /* Type of frame in particles buffer
         */
        typedef typename BufferType::FrameType FrameType;
        /* Type of border frame in a particle buffer
         */
        typedef typename BufferType::FrameTypeBorder FrameTypeBorder;

        /* Type of the particle box which particle buffer create
         */
        typedef typename BufferType::ParticlesBoxType ParticlesBoxType;

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
        BufferType* particlesBuffer;

        ParticlesBase(const std::shared_ptr<T_DeviceHeap>& deviceHeap, MappingDesc description)
            : SimulationFieldHelper<MappingDesc>(description)
            , particlesBuffer(NULL)
        {
            particlesBuffer = new BufferType(
                deviceHeap,
                description.getGridLayout().getDataSpace(),
                MappingDesc::SuperCellSize::toRT());
        }

        virtual ~ParticlesBase()
        {
            delete this->particlesBuffer;
        }

        /* Shift all particle in a AREA
         * @tparam AREA area which is used (CORE,BORDER,GUARD or a combination)
         */
        template<uint32_t AREA>
        void shiftParticles()
        {
            StrideMapping<AREA, 3, MappingDesc> mapper(this->cellDescription);
            ParticlesBoxType pBox = particlesBuffer->getDeviceParticleBox();

            constexpr uint32_t numWorkers
                = traits::GetNumWorkers<math::CT::volume<typename FrameType::SuperCellSize>::type::value>::value;
            __startTransaction(__getTransactionEvent());
            do
            {
                PMACC_KERNEL(KernelShiftParticles<numWorkers>{})
                (mapper.getGridDim(), numWorkers)(pBox, mapper);
            } while(mapper.next());

            __setTransactionEvent(__endTransaction());
        }

        /* fill gaps in a AREA
         * @tparam AREA area which is used (CORE,BORDER,GUARD or a combination)
         */
        template<uint32_t AREA>
        void fillGaps()
        {
            AreaMapping<AREA, MappingDesc> mapper(this->cellDescription);

            constexpr uint32_t numWorkers
                = traits::GetNumWorkers<math::CT::volume<typename FrameType::SuperCellSize>::type::value>::value;

            PMACC_KERNEL(KernelFillGaps<numWorkers>{})
            (mapper.getGridDim(), numWorkers)(particlesBuffer->getDeviceParticleBox(), mapper);
        }


    public:
        /* fill gaps in a the complete simulation area (include GUARD)
         */
        void fillAllGaps()
        {
            this->fillGaps<CORE + BORDER + GUARD>();
        }

        /* fill all gaps in the border of the simulation
         */
        void fillBorderGaps()
        {
            this->fillGaps<BORDER>();
        }

        /* Delete all particles in GUARD for one direction.
         */
        void deleteGuardParticles(uint32_t exchangeType);

        /* Delete all particle in an area*/
        template<uint32_t T_area>
        void deleteParticlesInArea();

        /** copy guard particles to intermediate exchange buffer
         *
         * Copy all particles from the guard of a direction to the device exchange buffer.
         * @warning This method resets the number of particles in the processed supercells even
         * if there are particles left in the supercell and does not guarantee that the last frame is
         * contiguous filled.
         * Call fillAllGaps afterwards if you need a valid number of particles
         * and a contiguously filled last frame.
         */
        void copyGuardToExchange(uint32_t exchangeType);

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
            PMACC_ASSERT(particlesBuffer != nullptr);
            return *particlesBuffer;
        }

        /* set all internal objects to initial state*/
        virtual void reset(uint32_t currentStep);
    };

} // namespace pmacc

#include "pmacc/particles/ParticlesBase.tpp"
