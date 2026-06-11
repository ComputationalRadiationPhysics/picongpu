/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/fields/SimulationFieldHelper.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"
#include "pmacc/mappings/kernel/ExchangeMapping.hpp"
#include "pmacc/particles/memory/boxes/ParticlesBox.hpp"
#include "pmacc/particles/memory/buffers/ParticlesBuffer.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

namespace pmacc
{
    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::deleteGuardParticles(uint32_t exchangeType)
    {
        ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);

        PMACC_LOCKSTEP_KERNEL(KernelDeleteParticles{})
            .config(mapper.getGridDim(), *particlesBuffer)(particlesBuffer->getDeviceParticleBox(), mapper);
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    template<uint32_t T_area>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::deleteParticlesInArea()
    {
        auto const mapper = makeAreaMapper<T_area>(this->cellDescription);

        PMACC_LOCKSTEP_KERNEL(KernelDeleteParticles{})
            .config(mapper.getGridDim(), *particlesBuffer)(particlesBuffer->getDeviceParticleBox(), mapper);
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

            particlesBuffer->getSendExchangeStack(exchangeType).setSize(0);

            PMACC_LOCKSTEP_KERNEL(KernelCopyGuardToExchange{})
                .config(mapper.getGridDim(), *particlesBuffer)(
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

                PMACC_LOCKSTEP_KERNEL(KernelInsertParticles{})
                    .config(numParticles, *particlesBuffer)(
                        particlesBuffer->getDeviceParticleBox(),
                        particlesBuffer->getReceiveExchangeStack(exchangeType).getDeviceExchangePopDataBox(),
                        mapper);
            }
        }
    }

} // namespace pmacc

#include "pmacc/particles/AsyncCommunicationImpl.hpp"
