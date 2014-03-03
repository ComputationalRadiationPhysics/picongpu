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

        GridController<DIM>& getGridController()
        {
            return GridController<DIM>::getInstance();
        }

        StreamController& getStreamController()
        {
            return StreamController::getInstance();
        }

        Manager& getManager()
        {
            return Manager::getInstance();
        }

        TransactionManager& getTransactionManager()
        {
            return TransactionManager::getInstance();
        }

        SubGrid<DIM>& getSubGrid()
        {
            return SubGrid<DIM>::getInstance();
        }

        EnvironmentController& getEnvironmentController()
        {
            return EnvironmentController::getInstance();
        }

        Factory& getFactory()
        {
            return Factory::getInstance();
        }
        
        ParticleFactory& getParticleFactory()
        {
            return ParticleFactory::getInstance();
        }
        
        DataConnector& getDataConnector()
        {
            return DataConnector::getInstance();
        }
        
        ModuleConnector& getModuleConnector()
        {
            return ModuleConnector::getInstance();
        }
        
        nvidia::memory::MemoryInfo& getEnvMemoryInfo()
        {
            return nvidia::memory::MemoryInfo::getInstance();
        }
        
        
        static Environment<DIM>& getInstance()
        {
            static Environment<DIM> instance;
            return instance;
        }
        
        void init(DataSpace<DIM> gridSize, DataSpace<DIM> devices, DataSpace<DIM> periodic)
        {
            GridController<DIM>::getInstance().init(devices, periodic);

            StreamController::getInstance();

            TransactionManager::getInstance();

            DataSpace<DIM> localGridSize(gridSize / devices);

            SubGrid<DIM>::getInstance().init(localGridSize, gridSize, GridController<DIM>::getInstance().getPosition() * localGridSize);

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

#define __startTransaction(...) (Environment<>::getInstance().getTransactionManager().startTransaction(__VA_ARGS__))
#define __startAtomicTransaction(...) (Environment<>::getInstance().getTransactionManager().startAtomicTransaction(__VA_ARGS__))
#define __endTransaction() (Environment<>::getInstance().getTransactionManager().endTransaction())
#define __startOperation(opType) (Environment<>::getInstance().getTransactionManager().startOperation(opType))
#define __getEventStream(opType) (Environment<>::getInstance().getTransactionManager().getEventStream(opType))
#define __getTransactionEvent() (Environment<>::getInstance().getTransactionManager().getTransactionEvent())
#define __setTransactionEvent(event) (Environment<>::getInstance().getTransactionManager().setTransactionEvent((event)))

}

#include "particles/tasks/ParticleFactory.tpp"
