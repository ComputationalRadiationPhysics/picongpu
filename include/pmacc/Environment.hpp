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

            /** get the singleton StreamController
             *
             * @return instance of StreamController
             */
            HINLINE pmacc::StreamController& StreamController();

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

        /** get the singleton Filesystem
         *
         * @return instance of Filesystem
         */
        HINLINE pmacc::Filesystem<T_dim>& Filesystem();

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

        Environment(const Environment&) = delete;

        Environment& operator=(const Environment&) = delete;

    private:
        Environment() = default;

        ~Environment() = default;
    };

} // namespace pmacc

#include "pmacc/Environment.tpp"
