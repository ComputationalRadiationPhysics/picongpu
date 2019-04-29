/* Copyright 2014-2019 Felix Schmitt, Conrad Schumann,
 *                     Alexander Grund, Axel Huebl
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

#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/particles/tasks/ParticleFactory.hpp"

#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/mappings/simulation/SubGrid.hpp"
#include "pmacc/mappings/simulation/EnvironmentController.hpp"
#include "pmacc/eventSystem/streams/StreamController.hpp"
#include "pmacc/dataManagement/DataConnector.hpp"
#include "pmacc/pluginSystem/PluginConnector.hpp"
#include "pmacc/nvidia/memory/MemoryInfo.hpp"
#include "pmacc/simulationControl/SimulationDescription.hpp"
#include "pmacc/mappings/simulation/Filesystem.hpp"
#include "pmacc/eventSystem/events/EventPool.hpp"
#include "pmacc/Environment.def"
#include "pmacc/communication/manager_common.hpp"
#include "pmacc/assert.hpp"

#include <mpi.h>

namespace pmacc
{

namespace detail
{
    /** collect state variables of the environment context
     *
     * This class handle the initialization and finalize of the
     * MPI context and the selection of the GPU.
     */
    class EnvironmentContext
    {

        friend Environment;

        friend pmacc::Environment<DIM1>;
        friend pmacc::Environment<DIM2>;
        friend pmacc::Environment<DIM3>;

        EnvironmentContext( ) :
            m_isMpiInitialized( false ),
            m_isDeviceSelected( false ),
            m_isSubGridDefined( false )
        {
        }

        /** initialization state of MPI */
        bool m_isMpiInitialized;

        /** state if a computing device is selected */
        bool m_isDeviceSelected;

        /** state if the SubGrid is defined */
        bool m_isSubGridDefined;

        /** get the singleton EnvironmentContext
         *
         * @return instance of EnvironmentContext
         */
        static EnvironmentContext& getInstance()
        {
            static EnvironmentContext instance;
            return instance;
        }

        /** state of the MPI context
         *
         * @return true if MPI is initialized else false
         */
        bool isMpiInitialized()
        {
            return m_isMpiInitialized;
        }

        /** is a computing device selected
         *
         * @return true if device is selected else false
         */
        bool isDeviceSelected()
        {
            return m_isDeviceSelected;
        }

        /** is the SubGrid defined
         *
         * @return true if SubGrid is defined, else false
         */
        bool isSubGridDefined()
        {
            return m_isSubGridDefined;
        }

        /** initialize the environment
         *
         * After this call it is allowed to use MPI.
         */
        void init();

        /** cleanup the environment */
        void finalize();

        /** select a computing device
         *
         * After this call it is allowed to use the computing device.
         *
         * @param deviceNumber number of the device
         */
        void setDevice(int deviceNumber);

    };

    /** PMacc environment
     *
     * Get access to all PMacc singleton classes those not depend on a dimension.
     */
    struct Environment
    {
        Environment()
        {
        }

        /** cleanup the environment */
        void finalize()
        {
            EnvironmentContext::getInstance().finalize();
        }

        /** get the singleton StreamController
         *
         * @return instance of StreamController
         */
        pmacc::StreamController& StreamController()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isDeviceSelected(),
                "Environment< DIM >::initDevices() must be called before this method!"
            );
            return StreamController::getInstance();
        }

        /** get the singleton Manager
         *
         * @return instance of Manager
         */
        pmacc::Manager& Manager()
        {
            return Manager::getInstance();
        }

        /** get the singleton TransactionManager
         *
         * @return instance of TransactionManager
         */
        pmacc::TransactionManager& TransactionManager() const
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isDeviceSelected(),
                "Environment< DIM >::initDevices() must be called before this method!"
            );
            return TransactionManager::getInstance();
        }

        /** get the singleton EnvironmentController
         *
         * @return instance of EnvironmentController
         */
        pmacc::EnvironmentController& EnvironmentController()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isMpiInitialized(),
                "Environment< DIM >::initDevices() must be called before this method!"
            );
            return EnvironmentController::getInstance();
        }

        /** get the singleton Factory
         *
         * @return instance of Factory
         */
        pmacc::Factory& Factory()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isMpiInitialized() &&
                EnvironmentContext::getInstance().isDeviceSelected(),
                "Environment< DIM >::initDevices() must be called before this method!"
            );
            return Factory::getInstance();
        }

        /** get the singleton EventPool
         *
         * @return instance of EventPool
         */
        pmacc::EventPool& EventPool()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isDeviceSelected(),
                "Environment< DIM >::initDevices() must be called before this method!"
            );
            return EventPool::getInstance();
        }

        /** get the singleton ParticleFactory
         *
         * @return instance of ParticleFactory
         */
        pmacc::ParticleFactory& ParticleFactory()
        {
            return ParticleFactory::getInstance();
        }

        /** get the singleton DataConnector
         *
         * @return instance of DataConnector
         */
        pmacc::DataConnector& DataConnector()
        {
            return DataConnector::getInstance();
        }

        /** get the singleton PluginConnector
         *
         * @return instance of PluginConnector
         */
        pmacc::PluginConnector& PluginConnector()
        {
            return PluginConnector::getInstance();
        }

        /** get the singleton MemoryInfo
         *
         * @return instance of MemoryInfo
         */
        nvidia::memory::MemoryInfo& MemoryInfo()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isDeviceSelected(),
                "Environment< DIM >::initDevices() must be called before this method!"
            );
            return nvidia::memory::MemoryInfo::getInstance();
        }

        /** get the singleton SimulationDescription
         *
         * @return instance of SimulationDescription
         */
        simulationControl::SimulationDescription& SimulationDescription()
        {
            return simulationControl::SimulationDescription::getInstance();
        }
    };
} // namespace detail

/** Global Environment singleton for PMacc
 */
template< uint32_t T_dim >
class Environment : public detail::Environment
{
public:

    /** get the singleton GridController
     *
     * @return instance of GridController
     */
    pmacc::GridController< T_dim >& GridController()
    {
        PMACC_ASSERT_MSG(
            detail::EnvironmentContext::getInstance().isMpiInitialized(),
            "Environment< DIM >::initDevices() must be called before this method!"
        );
        return pmacc::GridController< T_dim >::getInstance();
    }

    /** get the singleton SubGrid
     *
     * @return instance of SubGrid
     */
    pmacc::SubGrid< T_dim >& SubGrid()
    {
        PMACC_ASSERT_MSG(
            detail::EnvironmentContext::getInstance().isSubGridDefined(),
            "Environment< DIM >::initGrids() must be called before this method!"
        );
        return pmacc::SubGrid< T_dim >::getInstance();
    }

    /** get the singleton Filesystem
     *
     * @return instance of Filesystem
     */
    pmacc::Filesystem< T_dim >& Filesystem()
    {
        return pmacc::Filesystem< T_dim >::getInstance();
    }

    /** get the singleton Environment< DIM >
     *
     * @return instance of Environment<DIM >
     */
    static Environment< T_dim >& get()
    {
        static Environment< T_dim > instance;
        return instance;
    }

    /** create and initialize the environment of PMacc
     *
     * Usage of MPI or device(accelerator) function calls before this method
     * are not allowed.
     *
     * @param devices number of devices per simulation dimension
     * @param periodic periodicity each simulation dimension
     *                 (0 == not periodic, 1 == periodic)
     */
    void initDevices(
        DataSpace< T_dim > devices,
        DataSpace< T_dim > periodic
    )
    {
        // initialize the MPI context
        detail::EnvironmentContext::getInstance().init();

        // create singleton instances
        GridController().init( devices, periodic );

        EnvironmentController();

        Filesystem();

        detail::EnvironmentContext::getInstance().setDevice(
            static_cast<int>( GridController().getHostRank() )
        );

        StreamController().activate();

        MemoryInfo();

        TransactionManager();

        SimulationDescription();

    }

    /** initialize the computing domain information of PMacc
     *
     * @param globalDomainSize size of the global simulation domain [cells]
     * @param localDomainSize size of the local simulation domain [cells]
     * @param localDomainOffset local domain offset [cells]
     */
    void initGrids(
        DataSpace< T_dim > globalDomainSize,
        DataSpace< T_dim > localDomainSize,
        DataSpace< T_dim > localDomainOffset
    )
    {
        PMACC_ASSERT_MSG(
            detail::EnvironmentContext::getInstance().isMpiInitialized(),
            "Environment< DIM >::initDevices() must be called before this method!"
        );

        detail::EnvironmentContext::getInstance().m_isSubGridDefined = true;

        // create singleton instances
        SubGrid().init(
            localDomainSize,
            globalDomainSize,
            localDomainOffset
        );

        DataConnector();

        PluginConnector();
    }

    Environment(const Environment&) = delete;

    Environment& operator=(const Environment&) = delete;

private:

    Environment()
    {
    }

    ~Environment()
    {

    }

};

namespace detail
{

    void EnvironmentContext::init()
    {
        m_isMpiInitialized = true;

        // MPI_Init with NULL is allowed since MPI 2.0
        MPI_CHECK(MPI_Init(NULL,NULL));
    }

    void EnvironmentContext::finalize()
    {
        if( m_isMpiInitialized )
        {
            pmacc::Environment<>::get().Manager().waitForAllTasks();
            // Required by scorep for flushing the buffers
            cudaDeviceSynchronize();
            m_isMpiInitialized = false;
            /* Free the MPI context.
             * The gpu context is freed by the `StreamController`, because
             * MPI and CUDA are independent.
             */
            MPI_CHECK(MPI_Finalize());
        }
    }

    void EnvironmentContext::setDevice(int deviceNumber)
    {
        int num_gpus = 0; //number of gpus
        cudaGetDeviceCount(&num_gpus);
#if (PMACC_CUDA_ENABLED == 1)
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
#endif

        int maxTries = num_gpus;
#if (PMACC_CUDA_ENABLED == 1)
        cudaDeviceProp devProp;
        CUDA_CHECK((cuplaError_t)cudaGetDeviceProperties(&devProp, deviceNumber));
        /* if the gpu compute mode is set to default we use the given `deviceNumber` */
        if (devProp.computeMode == cudaComputeModeDefault)
            maxTries = 1;
#endif
        cudaError rc;

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
#if (PMACC_CUDA_ENABLED == 1)
                cudaDeviceProp dprop;
                CUDA_CHECK((cuplaError_t)cudaGetDeviceProperties(&dprop, tryDeviceId));
                log<ggLog::CUDA_RT > ("Set device to %1%: %2%") % tryDeviceId % dprop.name;
                if(cudaErrorSetOnActiveProcess == cudaSetDeviceFlags(cudaDeviceScheduleSpin))
                {
                    cudaGetLastError(); //reset all errors
                    /* - because of cudaStreamCreate was called cudaSetDeviceFlags crashed
                     * - to set the flags reset the device and set flags again
                     */
                    CUDA_CHECK(cudaDeviceReset());
                    CUDA_CHECK((cuplaError_t)cudaSetDeviceFlags(cudaDeviceScheduleSpin));
                }
#endif
                CUDA_CHECK(cudaGetLastError());
                break;
            }
            else if (rc == cudaErrorDeviceAlreadyInUse
#if (PMACC_CUDA_ENABLED == 1)
                || rc==(cudaError)cudaErrorDevicesUnavailable
#endif
            )
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

        m_isDeviceSelected = true;
    }

} // namespace detail
} // namespace pmacc

/* No namespace for macro defines */

/** start a dependency chain */
#define __startTransaction(...) (pmacc::Environment<>::get().TransactionManager().startTransaction(__VA_ARGS__))

/** end a opened dependency chain */
#define __endTransaction() (pmacc::Environment<>::get().TransactionManager().endTransaction())

/** mark the begin of an operation
 *
 * depended on the opType this method is blocking
 *
 * @param opType place were the operation is running
 *               possible places are: `ITask::TASK_CUDA`, `ITask::TASK_MPI`, `ITask::TASK_HOST`
 */
#define __startOperation(opType) (pmacc::Environment<>::get().TransactionManager().startOperation(opType))

/** get a `EventStream` that must be used for cuda calls
 *
 * depended on the opType this method is blocking
 *
 * @param opType place were the operation is running
 *               possible places are: `ITask::TASK_CUDA`, `ITask::TASK_MPI`, `ITask::TASK_HOST`
 */
#define __getEventStream(opType) (pmacc::Environment<>::get().TransactionManager().getEventStream(opType))

/** get the event of the current transaction */
#define __getTransactionEvent() (pmacc::Environment<>::get().TransactionManager().getTransactionEvent())

/** set a event to the current transaction */
#define __setTransactionEvent(event) (pmacc::Environment<>::get().TransactionManager().setTransactionEvent((event)))

#include "pmacc/eventSystem/EventSystem.tpp"
#include "pmacc/particles/tasks/ParticleFactory.tpp"
#include "pmacc/eventSystem/events/CudaEvent.hpp"
