/**
 * Copyright 2014 Felix Schmitt, Conrad Schumann
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
#include "particles/tasks/ParticleFactory.hpp"

#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "mappings/simulation/EnvironmentController.hpp"
#include "eventSystem/streams/StreamController.hpp"
#include "dataManagement/DataConnector.hpp"
#include "moduleSystem/ModuleConnector.hpp"
#include "nvidia/memory/MemoryInfo.hpp"



namespace PMacc
{

    template <unsigned DIM = DIM1>
    class Environment
    {
    public:

        PMacc::GridController<DIM>& GridController()
        {
            return PMacc::GridController<DIM>::getInstance();
        }

        StreamController& StreamController()
        {
            return StreamController::getInstance();
        }

        Manager& Manager()
        {
            return Manager::getInstance();
        }

        TransactionManager& TransactionManager()
        {
            return TransactionManager::getInstance();
        }

        PMacc::SubGrid<DIM>& SubGrid()
        {
            return PMacc::SubGrid<DIM>::getInstance();
        }

        EnvironmentController& EnvironmentController()
        {
            return EnvironmentController::getInstance();
        }

        Factory& Factory()
        {
            return Factory::getInstance();
        }
        
        ParticleFactory& ParticleFactory()
        {
            return ParticleFactory::getInstance();
        }
        
        DataConnector& DataConnector()
        {
            return DataConnector::getInstance();
        }
        
        ModuleConnector& ModuleConnector()
        {
            return ModuleConnector::getInstance();
        }
        
        nvidia::memory::MemoryInfo& EnvMemoryInfo()
        {
            return nvidia::memory::MemoryInfo::getInstance();
        }
        
        
        static Environment<DIM>& get()
        {
            static Environment<DIM> instance;
            return instance;
        }
        
        void init(DataSpace<DIM> gridSize, DataSpace<DIM> devices, DataSpace<DIM> periodic)
        {
            PMacc::GridController<DIM>::getInstance().init(devices, periodic);

            StreamController::getInstance().activate();

            TransactionManager::getInstance();

            DataSpace<DIM> localGridSize(gridSize / devices);

            PMacc::SubGrid<DIM>::getInstance().init(localGridSize, gridSize, PMacc::GridController<DIM>::getInstance().getPosition() * localGridSize);

            EnvironmentController::getInstance();
            
            DataConnector::getInstance();
            
            ModuleConnector::getInstance(); 
            
            nvidia::memory::MemoryInfo::getInstance();  
            
        }

        void deinit()
        {
        }

    private:

        Environment()
        {
        }

        Environment(const Environment&);

        Environment& operator=(const Environment&);

    };

#define __startTransaction(...) (Environment<>::get().TransactionManager().startTransaction(__VA_ARGS__))
#define __startAtomicTransaction(...) (Environment<>::get().TransactionManager().startAtomicTransaction(__VA_ARGS__))
#define __endTransaction() (Environment<>::get().TransactionManager().endTransaction())
#define __startOperation(opType) (Environment<>::get().TransactionManager().startOperation(opType))
#define __getEventStream(opType) (Environment<>::get().TransactionManager().getEventStream(opType))
#define __getTransactionEvent() (Environment<>::get().TransactionManager().getTransactionEvent())
#define __setTransactionEvent(event) (Environment<>::get().TransactionManager().setTransactionEvent((event)))

}

#include "particles/tasks/ParticleFactory.tpp"
