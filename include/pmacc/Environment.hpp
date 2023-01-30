/* Copyright 2014-2022 Felix Schmitt, Conrad Schumann,
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

#include "pmacc/Environment.def"
#include "pmacc/assert.hpp"
#include "pmacc/communication/manager_common.hpp"
#include "pmacc/dataManagement/DataConnector.hpp"
#include "pmacc/device/MemoryInfo.hpp"
#include "pmacc/eventSystem/eventSystem.hpp"
#include "pmacc/eventSystem/events/EventPool.hpp"
#include "pmacc/eventSystem/streams/StreamController.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/mappings/simulation/Filesystem.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/mappings/simulation/SubGrid.hpp"
#include "pmacc/particles/tasks/ParticleFactory.hpp"
#include "pmacc/pluginSystem/PluginConnector.hpp"
#include "pmacc/simulationControl/SimulationDescription.hpp"

#include <cstdlib> // std::getenv
#include <iostream>
#include <stdexcept> // std::invalid_argument

#include <mpi.h>

#if !defined(ALPAKA_API_PREFIX)
/* ALPAKA_API_PREFIX was removed in alpaka 1.0.0 but is required to get access cuda/hip functions directly.
 * @todo find a better way to access native cuda/Hip functions or try to avoid accessing these at all.
 */
#    include "pmacc/ppFunctions.hpp"
#    if(BOOST_LANG_CUDA)
#        define ALPAKA_API_PREFIX(name) PMACC_JOIN(cuda, name)
#    elif(BOOST_LANG_HIP)
#        define ALPAKA_API_PREFIX(name) PMACC_JOIN(hip, name)
#    endif
#endif

namespace pmacc
{
    namespace detail
    {
        /** PMacc environment
         *
         * Get access to all PMacc singleton classes those not depend on a dimension.
         */
        struct Environment
        {
            Environment() = default;

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
                    "Environment< DIM >::initDevices() must be called before this method!");
                return StreamController::getInstance();
            }

            /** get the singleton EnvironmentController
             *
             * @return instance of EnvironmentController
             */
            pmacc::EnvironmentController& EnvironmentController()
            {
                PMACC_ASSERT_MSG(
                    EnvironmentContext::getInstance().isMpiInitialized(),
                    "Environment< DIM >::initDevices() must be called before this method!");
                return EnvironmentController::getInstance();
            }

            /** get the singleton Factory
             *
             * @return instance of Factory
             */
            pmacc::Factory& Factory()
            {
                PMACC_ASSERT_MSG(
                    EnvironmentContext::getInstance().isMpiInitialized()
                        && EnvironmentContext::getInstance().isDeviceSelected(),
                    "Environment< DIM >::initDevices() must be called before this method!");
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
                    "Environment< DIM >::initDevices() must be called before this method!");
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
            device::MemoryInfo& MemoryInfo()
            {
                PMACC_ASSERT_MSG(
                    EnvironmentContext::getInstance().isDeviceSelected(),
                    "Environment< DIM >::initDevices() must be called before this method!");
                return device::MemoryInfo::getInstance();
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
    template<uint32_t T_dim>
    class Environment : public detail::Environment
    {
    public:
        void enableMpiDirect()
        {
            detail::EnvironmentContext::getInstance().enableMpiDirect();
        }

        bool isMpiDirectEnabled() const
        {
            return detail::EnvironmentContext::getInstance().isMpiDirectEnabled();
        }

        /** get the singleton GridController
         *
         * @return instance of GridController
         */
        pmacc::GridController<T_dim>& GridController()
        {
            PMACC_ASSERT_MSG(
                detail::EnvironmentContext::getInstance().isMpiInitialized(),
                "Environment< DIM >::initDevices() must be called before this method!");
            return pmacc::GridController<T_dim>::getInstance();
        }

        /** get the singleton SubGrid
         *
         * @return instance of SubGrid
         */
        pmacc::SubGrid<T_dim>& SubGrid()
        {
            PMACC_ASSERT_MSG(
                detail::EnvironmentContext::getInstance().isSubGridDefined(),
                "Environment< DIM >::initGrids() must be called before this method!");
            return pmacc::SubGrid<T_dim>::getInstance();
        }

        /** get the singleton Filesystem
         *
         * @return instance of Filesystem
         */
        pmacc::Filesystem<T_dim>& Filesystem()
        {
            return pmacc::Filesystem<T_dim>::getInstance();
        }

        /** get the singleton Environment< DIM >
         *
         * @return instance of Environment<DIM >
         */
        static Environment<T_dim>& get()
        {
            static Environment<T_dim> instance;
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
        void initDevices(DataSpace<T_dim> devices, DataSpace<T_dim> periodic)
        {
            // initialize the MPI context
            detail::EnvironmentContext::getInstance().init();

            // create singleton instances
            GridController().init(devices, periodic);

            EnvironmentController();

            Filesystem();

            detail::EnvironmentContext::getInstance().setDevice(static_cast<int>(GridController().getHostRank()));

            StreamController().activate();

            MemoryInfo();

            SimulationDescription();
        }

        /** initialize the computing domain information of PMacc
         *
         * @param globalDomainSize size of the global simulation domain [cells]
         * @param localDomainSize size of the local simulation domain [cells]
         * @param localDomainOffset local domain offset [cells]
         */
        void initGrids(
            DataSpace<T_dim> globalDomainSize,
            DataSpace<T_dim> localDomainSize,
            DataSpace<T_dim> localDomainOffset)
        {
            PMACC_ASSERT_MSG(
                detail::EnvironmentContext::getInstance().isMpiInitialized(),
                "Environment< DIM >::initDevices() must be called before this method!");

            detail::EnvironmentContext::getInstance().m_isSubGridDefined = true;

            // create singleton instances
            SubGrid().init(localDomainSize, globalDomainSize, localDomainOffset);

            DataConnector();

            PluginConnector();
        }

        Environment(const Environment&) = delete;

        Environment& operator=(const Environment&) = delete;

    private:
        Environment() = default;

        ~Environment() = default;
    };

    namespace detail
    {
        void EnvironmentContext::init()
        {
            m_isMpiInitialized = true;

            char const* env_value = std::getenv("PIC_USE_THREADED_MPI");
            if(env_value)
            {
                int required_level{};
                if(strcmp(env_value, "MPI_THREAD_SINGLE") == 0)
                {
                    required_level = MPI_THREAD_SINGLE;
                }
                else if(strcmp(env_value, "MPI_THREAD_FUNNELED") == 0)
                {
                    required_level = MPI_THREAD_FUNNELED;
                }
                else if(strcmp(env_value, "MPI_THREAD_SERIALIZED") == 0)
                {
                    required_level = MPI_THREAD_SERIALIZED;
                }
                else if(strcmp(env_value, "MPI_THREAD_MULTIPLE") == 0)
                {
                    required_level = MPI_THREAD_MULTIPLE;
                }
                else
                {
                    throw std::runtime_error(
                        "Environment variable PIC_USE_THREADED_MPI must be one of MPI_THREAD_SINGLE, "
                        "MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE.");
                }
                // MPI_Init with NULL is allowed since MPI 2.0
                MPI_CHECK(MPI_Init_thread(nullptr, nullptr, required_level, nullptr));
            }
            else
            {
                // MPI_Init with NULL is allowed since MPI 2.0
                MPI_CHECK(MPI_Init(nullptr, nullptr));
            }
        }

        void EnvironmentContext::finalize()
        {
            if(m_isMpiInitialized)
            {
                eventSystem::waitForAllTasks();
                // Required by scorep for flushing the buffers
                cuplaDeviceSynchronize();
                m_isMpiInitialized = false;
                /*
                 * When using the combination of MPI_Comm_Accept() and MPI_Open_Port() (as is done by the MPI-based
                 * implementation of the ADIOS2 SST engine), the current (2023-01-06) Cray MPI implementation
                 * on Crusher/Frontier will hang inside MPI_Finalize().
                 * The workaround is to replace it with an MPI_Barrier().
                 */
                char const* env_value = std::getenv("PIC_WORKAROUND_CRAY_MPI_FINALIZE");
                bool use_cray_workaround = false;
                if(env_value)
                {
                    try
                    {
                        int env_value_int = std::stoi(env_value);
                        use_cray_workaround = env_value_int != 0;
                    }
                    catch(std::invalid_argument const& e)
                    {
                        std::cerr
                            << "Warning: PIC_WORKAROUND_CRAY_MPI_FINALIZE must have an integer value, received: '"
                            << env_value << "'. Will ignore." << std::endl;
                    }
                }
                /* Free the MPI context.
                 * The gpu context is freed by the `StreamController`, because
                 * MPI and CUDA are independent.
                 */
                if(use_cray_workaround)
                {
                    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                }
                else
                {
                    MPI_CHECK(MPI_Finalize());
                }
            }
        }

        void EnvironmentContext::setDevice(int deviceNumber)
        {
            int num_gpus = 0; // number of gpus
            cuplaGetDeviceCount(&num_gpus);
#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
            //##ERROR handling
            if(num_gpus < 1) // check if cupla device is found
            {
                throw std::runtime_error("no CUDA capable devices detected");
            }
#endif

            int maxTries = num_gpus;
            bool deviceSelectionSuccessful = false;

            cuplaError rc;

            // search the first selectable device in the compute node
            for(int deviceOffset = 0; deviceOffset < maxTries; ++deviceOffset)
            {
                /* Modulo 'num_gpus' avoids invalid device indices for systems where the environment variable
                 * `CUDA_VISIBLE_DEVICES` is used to pre-select a device.
                 */
                const int tryDeviceId = (deviceOffset + deviceNumber) % num_gpus;

                log<ggLog::CUDA_RT>("Trying to allocate device %1%.") % tryDeviceId;

#if(BOOST_LANG_CUDA || BOOST_LANG_HIP)
#    if(BOOST_LANG_CUDA)
                cudaDeviceProp devProp;
#    elif(BOOST_LANG_HIP)
                hipDeviceProp_t devProp;
#    endif

                CUDA_CHECK((cuplaError_t) ALPAKA_API_PREFIX(GetDeviceProperties)(&devProp, tryDeviceId));

                /* If the cuda gpu compute mode is 'default'
                 * (https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-modes)
                 * then we try to get a device only once.
                 * The index used to select a device is based on the local MPI rank so
                 * that each rank tries a different device.
                 */
                if(devProp.computeMode == ALPAKA_API_PREFIX(ComputeModeDefault))
                {
                    maxTries = 1;
                    log<ggLog::CUDA_RT>("Device %1% is running in default mode.") % tryDeviceId;
                }
#endif

                rc = cuplaSetDevice(tryDeviceId);

                if(rc == cuplaSuccess)
                {
                    cuplaStream_t stream;
                    /* \todo: Check if this workaround is needed
                     *
                     * - since NVIDIA change something in driver cuplaSetDevice never
                     * return an error if another process already use the selected
                     * device if gpu compute mode is set "process exclusive"
                     * - create a dummy stream to check if the device is already used by
                     * an other process.
                     * - cuplaStreamCreate fails if gpu is already in use
                     */
                    rc = cuplaStreamCreate(&stream);
                }

                if(rc == cuplaSuccess)
                {
#if(BOOST_LANG_CUDA || BOOST_LANG_HIP)
                    CUDA_CHECK((cuplaError_t) ALPAKA_API_PREFIX(GetDeviceProperties)(&devProp, tryDeviceId));
                    log<ggLog::CUDA_RT>("Set device to %1%: %2%") % tryDeviceId % devProp.name;
                    if(ALPAKA_API_PREFIX(ErrorSetOnActiveProcess)
                       == ALPAKA_API_PREFIX(SetDeviceFlags)(ALPAKA_API_PREFIX(DeviceScheduleSpin)))
                    {
                        cuplaGetLastError(); // reset all errors
                        /* - because of cuplaStreamCreate was called cuplaSetDeviceFlags crashed
                         * - to set the flags reset the device and set flags again
                         */
                        CUDA_CHECK(cuplaDeviceReset());
                        CUDA_CHECK(
                            (cuplaError_t) ALPAKA_API_PREFIX(SetDeviceFlags)(ALPAKA_API_PREFIX(DeviceScheduleSpin)));
                    }
#endif
                    CUDA_CHECK(cuplaGetLastError());
                    deviceSelectionSuccessful = true;
                    break;
                }
                else if(
                    rc == cuplaErrorDeviceAlreadyInUse
#if(PMACC_CUDA_ENABLED == 1)
                    || rc == (cuplaError) cudaErrorDevicesUnavailable
#endif
                )
                {
                    cuplaGetLastError(); // reset all errors
                    log<ggLog::CUDA_RT>("Device %1% already in use, try next.") % tryDeviceId;
                    continue;
                }
                else
                {
                    CUDA_CHECK(rc); /*error message*/
                }
            }
            if(!deviceSelectionSuccessful)
            {
                std::cerr << "Failed to select one of the " << num_gpus << " devices." << std::endl;
                throw std::runtime_error("Compute device selection failed.");
            }

            m_isDeviceSelected = true;
        }

    } // namespace detail
} // namespace pmacc

/* No namespace for macro defines */

#include "pmacc/eventSystem/events/CudaEvent.hpp"
#include "pmacc/eventSystem/tasks/Factory.tpp"
#include "pmacc/particles/tasks/ParticleFactory.tpp"
