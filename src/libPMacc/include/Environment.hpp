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

#include "eventSystem/EventSystem.hpp"
#include "particles/tasks/ParticleFactory.hpp"

#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "mappings/simulation/EnvironmentController.hpp"
#include "eventSystem/streams/StreamController.hpp"
#include "dataManagement/DataConnector.hpp"
#include "pluginSystem/PluginConnector.hpp"
#include "nvidia/memory/MemoryInfo.hpp"
#include "mappings/simulation/Filesystem.hpp"


namespace PMacc
{

    /**
     * Global Environment singleton for Picongpu
     */

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

        TransactionManager& TransactionManager() const
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

        PluginConnector& PluginConnector()
        {
            return PluginConnector::getInstance();
        }

        nvidia::memory::MemoryInfo& EnvMemoryInfo()
        {
            return nvidia::memory::MemoryInfo::getInstance();
        }

        PMacc::Filesystem<DIM>& Filesystem()
        {
            return PMacc::Filesystem<DIM>::getInstance();
        }

        static Environment<DIM>& get()
        {
            static Environment<DIM> instance;
            return instance;
        }

        void initDevices(DataSpace<DIM> devices, DataSpace<DIM> periodic)
        {
            PMacc::GridController<DIM>::getInstance().init(devices, periodic);

            PMacc::Filesystem<DIM>::getInstance();

            setDevice((int) (PMacc::GridController<DIM>::getInstance().getHostRank()));

            StreamController::getInstance().activate();

            TransactionManager::getInstance();

        }

        void initGrids(DataSpace<DIM> gridSizeGlobal, DataSpace<DIM> gridSizeLocal, DataSpace<DIM> gridOffset)
        {
            PMacc::SubGrid<DIM>::getInstance().init(gridSizeLocal, gridSizeGlobal, gridOffset);

            EnvironmentController::getInstance();

            DataConnector::getInstance();

            PluginConnector::getInstance();

            nvidia::memory::MemoryInfo::getInstance();

        }

        void finalize()
        {
        }

    private:

        Environment()
        {
        }

        Environment(const Environment&);

        Environment& operator=(const Environment&);

        void setDevice(int deviceNumber)
        {
            int num_gpus = 0; //number of gpus
            cudaGetDeviceCount(&num_gpus);
            //##ERROR handling
            if (num_gpus < 1) //check if cuda device is found
            {
                throw std::runtime_error("no CUDA capable devices detected");
            }
            else if (num_gpus < deviceNumber) //check if device can be selected by deviceNumber
            {
                std::cerr << "no CUDA device " << deviceNumber << ", only " << num_gpus << " devices found" << std::endl;
                throw std::runtime_error("CUDA capable devices can't be selected");
            }

            cudaDeviceProp devProp;
            cudaError rc;
            CUDA_CHECK(cudaGetDeviceProperties(&devProp, deviceNumber));
            if (devProp.computeMode == cudaComputeModeDefault)
            {
                CUDA_CHECK(rc = cudaSetDevice(deviceNumber));
                if (cudaSuccess == rc)
                {
                    cudaDeviceProp dprop;
                    cudaGetDeviceProperties(&dprop, deviceNumber);
                    log<ggLog::CUDA_RT > ("Set device to %1%: %2%") % deviceNumber % dprop.name;
                }
            }
            else
            {
                //gpu mode is cudaComputeModeExclusiveProcess and a free device is automatically selected
                log<ggLog::CUDA_RT > ("Device is selected by CUDA automatically (since cudaComputeModeDefault is not set).");
            }
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

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
