/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include "pmacc/Environment.hpp"
#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

#include "pmacc/fields/SimulationFieldHelper.hpp"
#include "pmacc/mappings/kernel/ExchangeMapping.hpp"

#include "pmacc/particles/memory/boxes/ParticlesBox.hpp"
#include "pmacc/particles/memory/buffers/ParticlesBuffer.hpp"


namespace pmacc
{
    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::deleteGuardParticles(uint32_t exchangeType)
    {
        ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);

        constexpr uint32_t numWorkers
            = traits::GetNumWorkers<math::CT::volume<typename FrameType::SuperCellSize>::type::value>::value;

        PMACC_KERNEL(KernelDeleteParticles<numWorkers>{})
        (mapper.getGridDim(), numWorkers)(particlesBuffer->getDeviceParticleBox(), mapper);
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    template<uint32_t T_area>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::deleteParticlesInArea()
    {
        AreaMapping<T_area, MappingDesc> mapper(this->cellDescription);

        constexpr uint32_t numWorkers
            = traits::GetNumWorkers<math::CT::volume<typename FrameType::SuperCellSize>::type::value>::value;

        PMACC_KERNEL(KernelDeleteParticles<numWorkers>{})
        (mapper.getGridDim(), numWorkers)(particlesBuffer->getDeviceParticleBox(), mapper);
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::reset(uint32_t)
    {
        deleteParticlesInArea<CORE + BORDER + GUARD>();
        particlesBuffer->reset();
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::copyGuardToExchange(uint32_t exchangeType)
    {
        if(particlesBuffer->hasSendExchange(exchangeType))
        {
            ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);

            particlesBuffer->getSendExchangeStack(exchangeType).setCurrentSize(0);

            constexpr uint32_t numWorkers
                = traits::GetNumWorkers<math::CT::volume<typename FrameType::SuperCellSize>::type::value>::value;

            PMACC_KERNEL(KernelCopyGuardToExchange<numWorkers>{})
            (mapper.getGridDim(), numWorkers)(
                particlesBuffer->getDeviceParticleBox(),
                particlesBuffer->getSendExchangeStack(exchangeType).getDeviceExchangePushDataBox(),
                mapper);
        }
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::insertParticles(uint32_t exchangeType)
    {
        if(particlesBuffer->hasReceiveExchange(exchangeType))
        {
            size_t numParticles = 0u;
            if(Environment<>::get().isMpiDirectEnabled())
                numParticles = particlesBuffer->getReceiveExchangeStack(exchangeType).getDeviceCurrentSize();
            else
                numParticles = particlesBuffer->getReceiveExchangeStack(exchangeType).getHostCurrentSize();

            if(numParticles != 0u)
            {
                ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);

                constexpr uint32_t numWorkers
                    = traits::GetNumWorkers<math::CT::volume<typename FrameType::SuperCellSize>::type::value>::value;

                PMACC_KERNEL(KernelInsertParticles<numWorkers>{})
                (numParticles, numWorkers)(
                    particlesBuffer->getDeviceParticleBox(),
                    particlesBuffer->getReceiveExchangeStack(exchangeType).getDeviceExchangePopDataBox(),
                    mapper);
            }
        }
    }

} // namespace pmacc

#include "pmacc/particles/AsyncCommunicationImpl.hpp"
