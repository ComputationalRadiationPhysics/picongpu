/**
 * Copyright 2013 Heiko Burau, René Widera
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
 
/* 
 * File:   ParticlesBase.hpp
 * Author: widera
 *
 * Created on 17. Januar 2011, 08:33
 */



#include "eventSystem/EventSystem.hpp"

#include "fields/SimulationFieldHelper.hpp"
#include "mappings/kernel/ExchangeMapping.hpp"

#include "particles/tasks/ParticleFactory.hpp"

#include "particles/memory/boxes/ParticlesBox.hpp"
#include "particles/memory/buffers/ParticlesBuffer.hpp"


namespace PMacc
{

    template<typename PositionType,class UserTypeList, class MappingDesc>
    void ParticlesBase<PositionType,UserTypeList, MappingDesc>::bashParticles(uint32_t exchangeType)
    {
        if (particlesBuffer->hasSendExchange(exchangeType))
        {
            //std::cout<<"bash "<<exchangeType<<std::endl;
            ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);

            particlesBuffer->getSendExchangeStack(exchangeType).setCurrentSize(0);
            dim3 grid(mapper.getGridDim());

            __cudaKernel(kernelBashParticles)
                    (grid, TileSize)
                    (particlesBuffer->getDeviceParticleBox(),
                    particlesBuffer->getSendExchangeStack(exchangeType).getDeviceExchangePushDataBox(), mapper); 
        }
    }

    template<typename PositionType,class UserTypeList, class MappingDesc>
    void ParticlesBase<PositionType,UserTypeList, MappingDesc>::insertParticles(uint32_t exchangeType)
    {
        if (particlesBuffer->hasReceiveExchange(exchangeType))
        {

            dim3 grid(particlesBuffer->getReceiveExchangeStack(exchangeType).getHostCurrentSize());
            if (grid.x != 0)
            {
              //  std::cout<<"insert = "<<grid.x()<<std::endl;
                ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);
                __cudaKernel(kernelInsertParticles)
                        (grid, TileSize)
                        (particlesBuffer->getDeviceParticleBox(),
                        particlesBuffer->getReceiveExchangeStack(exchangeType).getDeviceExchangePopDataBox(),
                        mapper);
                }
        }
    }

    template<typename PositionType,class UserTypeList, class MappingDesc>
    EventTask ParticlesBase<PositionType,UserTypeList, MappingDesc>::asyncCommunication(EventTask event)
    {
        EventTask ret;
        __startTransaction(event);
        ParticleFactory::getInstance().createTaskParticlesReceive(*this);
        ret = __endTransaction();

        __startTransaction(event);
        ParticleFactory::getInstance().createTaskParticlesSend(*this);
        ret += __endTransaction();
        return ret;
    }

} //namespace PMacc


