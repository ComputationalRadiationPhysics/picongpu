/* Copyright 2014-2023 Felix Schmitt, Conrad Schumann,
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

#include "pmacc/Environment.hpp"
#include "pmacc/alpakaHelper/Device.hpp"
#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/communication/manager_common.hpp"
#include "pmacc/types.hpp"

#include <stdexcept>


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
        pmacc::StreamController& Environment::StreamController()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isDeviceSelected(),
                "Environment< DIM >::initDevices() must be called before this method!");
            return StreamController::getInstance();
        }

        pmacc::EnvironmentController& Environment::EnvironmentController()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isMpiInitialized(),
                "Environment< DIM >::initDevices() must be called before this method!");
            return EnvironmentController::getInstance();
        }


        pmacc::Factory& Environment::Factory()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isMpiInitialized()
                    && EnvironmentContext::getInstance().isDeviceSelected(),
                "Environment< DIM >::initDevices() must be called before this method!");
            return Factory::getInstance();
        }

        pmacc::EventPool& Environment::EventPool()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isDeviceSelected(),
                "Environment< DIM >::initDevices() must be called before this method!");
            return EventPool::getInstance();
        }


        pmacc::ParticleFactory& Environment::ParticleFactory()
        {
            return ParticleFactory::getInstance();
        }


        pmacc::DataConnector& Environment::DataConnector()
        {
            return DataConnector::getInstance();
        }


        pmacc::PluginConnector& Environment::PluginConnector()
        {
            return PluginConnector::getInstance();
        }

        device::MemoryInfo& Environment::MemoryInfo()
        {
            PMACC_ASSERT_MSG(
                EnvironmentContext::getInstance().isDeviceSelected(),
                "Environment< DIM >::initDevices() must be called before this method!");
            return device::MemoryInfo::getInstance();
        }


        simulationControl::SimulationDescription& Environment::SimulationDescription()
        {
            return simulationControl::SimulationDescription::getInstance();
        }

    } // namespace detail

    template<uint32_t T_dim>
    void Environment<T_dim>::enableMpiDirect()
    {
        detail::EnvironmentContext::getInstance().enableMpiDirect();
    }

    template<uint32_t T_dim>
    bool Environment<T_dim>::isMpiDirectEnabled() const
    {
        return detail::EnvironmentContext::getInstance().isMpiDirectEnabled();
    }

    template<uint32_t T_dim>
    pmacc::GridController<T_dim>& Environment<T_dim>::GridController()
    {
        PMACC_ASSERT_MSG(
            detail::EnvironmentContext::getInstance().isMpiInitialized(),
            "Environment< DIM >::initDevices() must be called before this method!");
        return pmacc::GridController<T_dim>::getInstance();
    }

    template<uint32_t T_dim>
    pmacc::SubGrid<T_dim>& Environment<T_dim>::SubGrid()
    {
        PMACC_ASSERT_MSG(
            detail::EnvironmentContext::getInstance().isSubGridDefined(),
            "Environment< DIM >::initGrids() must be called before this method!");
        return pmacc::SubGrid<T_dim>::getInstance();
    }

    template<uint32_t T_dim>
    pmacc::Filesystem<T_dim>& Environment<T_dim>::Filesystem()
    {
        return pmacc::Filesystem<T_dim>::getInstance();
    }


    template<uint32_t T_dim>
    void Environment<T_dim>::initDevices(DataSpace<T_dim> devices, DataSpace<T_dim> periodic)
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

    template<uint32_t T_dim>
    void Environment<T_dim>::initGrids(
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
                alpaka::wait(manager::Device<ComputeDevice>::get().current());
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
            int num_gpus = manager::Device<ComputeDevice>::get().count();

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
            // ##ERROR handling
            if(num_gpus < 1) // check if cupla device is found
            {
                throw std::runtime_error("no CUDA capable devices detected");
            }
#endif

            int maxTries = num_gpus;
            bool deviceSelectionSuccessful = false;

            // search the first selectable device in the compute node
            for(int deviceOffset = 0; deviceOffset < maxTries; ++deviceOffset)
            {
                // true if an error happened, else false
                bool error = false;

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

                auto err = ALPAKA_API_PREFIX(GetDeviceProperties)(&devProp, tryDeviceId);
                if(err != ALPAKA_API_PREFIX(Success))
                    throw std::runtime_error("Error reading device properties.");

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

                try
                {
                    manager::Device<ComputeDevice>::get().device(tryDeviceId);
                }
                catch(const std::system_error& e)
                {
                    error = true;
                }

                if(!error)
                {
                    /* Create a dummy stream to check if the device is already used by another process. This could
                     * happen on NVIDIA devices. alpaka is performing the same check during the device selection but
                     * not for all device types. This is a safty check if alpaka is not performing this check.
                     */
                    try
                    {
                        auto testStream = AccStream(manager::Device<ComputeDevice>::get().current());
                    }
                    catch(const std::system_error& e)
                    {
                        error = true;
                    }
                }

                if(!error)
                {
                    deviceSelectionSuccessful = true;

                    break;
                }
                else
                {
                    log<ggLog::CUDA_RT>("Device %1% already in use, try next.") % tryDeviceId;
                    continue;
                }
            }
            if(!deviceSelectionSuccessful)
            {
                std::cerr << "Failed to select one of the " << num_gpus << " devices." << std::endl;
                throw std::runtime_error("Compute device selection failed.");
            }

            // initialize the default host device
            manager::Device<HostDevice>::get().device();
            m_isDeviceSelected = true;
        }

    } // namespace detail
} // namespace pmacc

/* PMACC_NO_TPP_INCLUDE is only defined if pmaccHeaderCheck is running.
 * In this case the current tested hpp file can include a file which depend on the tested hpp file.
 * This will build a cyclic include we can only solve if we write everywhere clean header files without
 * implementations.
 */
#if !defined(PMACC_NO_TPP_INCLUDE)
#    include "pmacc/eventSystem/tasks/Factory.tpp"
#    include "pmacc/fields/tasks/FieldFactory.tpp"
#    include "pmacc/particles/tasks/ParticleFactory.tpp"
#endif
