/*
 * SPDX-FileCopyrightText: Felix Schmitt, Conrad Schumann, Alexander Grund, Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.def"
#include "pmacc/assert.hpp"
#include "pmacc/communication/manager_common.hpp"
#include "pmacc/dataManagement/DataConnector.hpp"
#include "pmacc/device/MemoryInfo.hpp"
#include "pmacc/eventSystem/eventSystem.hpp"
#include "pmacc/eventSystem/events/EventPool.hpp"
#include "pmacc/eventSystem/queues/QueueController.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/mappings/simulation/SubGrid.hpp"
#include "pmacc/particles/tasks/ParticleFactory.hpp"
#include "pmacc/pluginSystem/PluginConnector.hpp"
#include "pmacc/simulationControl/SimulationDescription.hpp"

#include <mpi.h>

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

            /** get the singleton QueueController
             *
             * @return instance of QueueController
             */
            HINLINE pmacc::QueueController& QueueController();

            /** get the singleton EnvironmentController
             *
             * @return instance of EnvironmentController
             */
            HINLINE pmacc::EnvironmentController& EnvironmentController();

            /** get the singleton Factory
             *
             * @return instance of Factory
             */
            HINLINE pmacc::Factory& Factory();

            /** get the singleton EventPool
             *
             * @return instance of EventPool
             */
            HINLINE pmacc::EventPool& EventPool();

            /** get the singleton ParticleFactory
             *
             * @return instance of ParticleFactory
             */
            HINLINE pmacc::ParticleFactory& ParticleFactory();

            /** get the singleton DataConnector
             *
             * @return instance of DataConnector
             */
            HINLINE pmacc::DataConnector& DataConnector();

            /** get the singleton PluginConnector
             *
             * @return instance of PluginConnector
             */
            HINLINE pmacc::PluginConnector& PluginConnector();

            /** get the singleton MemoryInfo
             *
             * @return instance of MemoryInfo
             */
            HINLINE device::MemoryInfo& MemoryInfo();

            /** get the singleton SimulationDescription
             *
             * @return instance of SimulationDescription
             */
            HINLINE simulationControl::SimulationDescription& SimulationDescription();
        };
    } // namespace detail

    /** Global Environment singleton for PMacc
     */
    template<uint32_t T_dim>
    class Environment : public detail::Environment
    {
    public:
        HINLINE void enableMpiDirect();

        HINLINE bool isMpiDirectEnabled() const;

        /** get the singleton GridController
         *
         * @return instance of GridController
         */
        HINLINE pmacc::GridController<T_dim>& GridController();

        /** get the singleton SubGrid
         *
         * @return instance of SubGrid
         */
        HINLINE pmacc::SubGrid<T_dim>& SubGrid();

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
        HINLINE void initDevices(DataSpace<T_dim> devices, DataSpace<T_dim> periodic);

        /** initialize the computing domain information of PMacc
         *
         * @param globalDomainSize size of the global simulation domain [cells]
         * @param localDomainSize size of the local simulation domain [cells]
         * @param localDomainOffset local domain offset [cells]
         */
        HINLINE void initGrids(
            DataSpace<T_dim> globalDomainSize,
            DataSpace<T_dim> localDomainSize,
            DataSpace<T_dim> localDomainOffset);

        Environment(Environment const&) = delete;

        Environment& operator=(Environment const&) = delete;

    private:
        Environment() = default;

        ~Environment() = default;
    };

} // namespace pmacc

#include "pmacc/Environment.tpp"
