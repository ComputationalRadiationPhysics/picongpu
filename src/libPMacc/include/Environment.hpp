/**
 * Copyright 2014-2016 Felix Schmitt, Conrad Schumann,
 *                     Alexander Grund, Axel Huebl
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

#include "eventSystem/EventSystem.hpp"
#include "particles/tasks/ParticleFactory.hpp"

#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "mappings/simulation/EnvironmentController.hpp"
#include "eventSystem/streams/StreamController.hpp"
#include "dataManagement/DataConnector.hpp"
#include "pluginSystem/PluginConnector.hpp"
#include "nvidia/memory/MemoryInfo.hpp"
#include "simulationControl/SimulationDescription.hpp"
#include "mappings/simulation/Filesystem.hpp"
#include "eventSystem/events/EventPool.hpp"
#include "Environment.def"

#include <cuda_runtime.h>

namespace PMacc
{

/**
 * Global Environment singleton for Picongpu
 */

template <unsigned DIM>
class Environment
{
public:

    PMacc::GridController<DIM>& GridController()
    {
        return PMacc::GridController<DIM>::getInstance();
    }

    PMacc::StreamController& StreamController()
    {
        return StreamController::getInstance();
    }

    PMacc::Manager& Manager()
    {
        return Manager::getInstance();
    }

    PMacc::TransactionManager& TransactionManager() const
    {
        return TransactionManager::getInstance();
    }

    PMacc::SubGrid<DIM>& SubGrid()
    {
        return PMacc::SubGrid<DIM>::getInstance();
    }

    PMacc::EnvironmentController& EnvironmentController()
    {
        return EnvironmentController::getInstance();
    }

    PMacc::EventPool& EventPool()
    {
        return EventPool::getInstance();
    }

    PMacc::Factory& Factory()
    {
        return Factory::getInstance();
    }

    PMacc::ParticleFactory& ParticleFactory()
    {
        return ParticleFactory::getInstance();
    }

    PMacc::DataConnector& DataConnector()
    {
        return DataConnector::getInstance();
    }

    PMacc::PluginConnector& PluginConnector()
    {
        return PluginConnector::getInstance();
    }

    nvidia::memory::MemoryInfo& MemoryInfo()
    {
        return nvidia::memory::MemoryInfo::getInstance();
    }

    simulationControl::SimulationDescription& SimulationDescription()
    {
        return simulationControl::SimulationDescription::getInstance();
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

        simulationControl::SimulationDescription::getInstance();
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
        else if (deviceNumber >= num_gpus) //check if device can be selected by deviceNumber
        {
            std::cerr << "no CUDA device " << deviceNumber << ", only " << num_gpus << " devices found" << std::endl;
            throw std::runtime_error("CUDA capable devices can't be selected");
        }


        int maxTries = num_gpus;

        cudaDeviceProp devProp;
        cudaError rc;
        CUDA_CHECK(cudaGetDeviceProperties(&devProp, deviceNumber));

        /* if the gpu compute mode is set to default we use the given `deviceNumber` */
        if (devProp.computeMode == cudaComputeModeDefault)
            maxTries = 1;

        for (int deviceOffset = 0; deviceOffset < maxTries; ++deviceOffset)
        {
            const int tryDeviceId = (deviceOffset + deviceNumber) % num_gpus;
            rc = cudaSetDevice(tryDeviceId);

            if(rc == cudaSuccess)
            {
               cudaStream_t stream;
               /* \todo: Check if this workaround is needed
                *
                * - since NVIDIA change something in driver cudaSetDevice never
                * return an error if another process already use the selected
                * device if gpu compute mode is set "process exclusive"
                * - create a dummy stream to check if the device is already used by
                * an other process.
                * - cudaStreamCreate fail if gpu is already in use
                */
               rc = cudaStreamCreate(&stream);
            }

            if (rc == cudaSuccess)
            {
                cudaDeviceProp dprop;
                CUDA_CHECK(cudaGetDeviceProperties(&dprop, tryDeviceId));
                log<ggLog::CUDA_RT > ("Set device to %1%: %2%") % tryDeviceId % dprop.name;
                if(cudaErrorSetOnActiveProcess == cudaSetDeviceFlags(cudaDeviceScheduleSpin))
                {
                    cudaGetLastError(); //reset all errors
                    /* - because of cudaStreamCreate was called cudaSetDeviceFlags crashed
                     * - to set the flags reset the device and set flags again
                     */
                    CUDA_CHECK(cudaDeviceReset());
                    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
                }
                CUDA_CHECK(cudaGetLastError());
                break;
            }
            else if (rc == cudaErrorDeviceAlreadyInUse || rc==cudaErrorDevicesUnavailable)
            {
                cudaGetLastError(); //reset all errors
                log<ggLog::CUDA_RT > ("Device %1% already in use, try next.") % tryDeviceId;
                continue;
            }
            else
            {
                CUDA_CHECK(rc); /*error message*/
            }
        }
    }
};

}

/* No namespace for macro defines */

/** start a dependency chain */
#define __startTransaction(...) (PMacc::Environment<>::get().TransactionManager().startTransaction(__VA_ARGS__))

/** end a opened dependency chain */
#define __endTransaction() (PMacc::Environment<>::get().TransactionManager().endTransaction())

/** mark the begin of an operation
 *
 * depended on the opType this method is blocking
 *
 * @param opType place were the operation is running
 *               possible places are: `ITask::TASK_CUDA`, `ITask::TASK_MPI`, `ITask::TASK_HOST`
 */
#define __startOperation(opType) (PMacc::Environment<>::get().TransactionManager().startOperation(opType))

/** get a `EventStream` that must be used for cuda calls
 *
 * depended on the opType this method is blocking
 *
 * @param opType place were the operation is running
 *               possible places are: `ITask::TASK_CUDA`, `ITask::TASK_MPI`, `ITask::TASK_HOST`
 */
#define __getEventStream(opType) (PMacc::Environment<>::get().TransactionManager().getEventStream(opType))

/** get the event of the current transaction */
#define __getTransactionEvent() (PMacc::Environment<>::get().TransactionManager().getTransactionEvent())

/** set a event to the current transaction */
#define __setTransactionEvent(event) (PMacc::Environment<>::get().TransactionManager().setTransactionEvent((event)))

#include "eventSystem/EventSystem.tpp"
#include "particles/tasks/ParticleFactory.tpp"
#include "eventSystem/events/CudaEvent.hpp"
