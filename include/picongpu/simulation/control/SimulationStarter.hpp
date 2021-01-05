/* Copyright 2013-2021 Axel Huebl, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <boost/program_options/options_description.hpp>
#include <iostream>

#include "picongpu/simulation_defines.hpp"
#include "picongpu/ArgsParser.hpp"
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/pluginSystem/PluginConnector.hpp>
#include "picongpu/simulation/control/ISimulationStarter.hpp"

namespace picongpu
{
    using namespace pmacc;

    template<class InitClass, class PluginClass, class SimulationClass>
    class SimulationStarter : public ISimulationStarter
    {
    private:
        using BoostOptionsList = std::list<boost::program_options::options_description>;

        SimulationClass* simulationClass;
        InitClass* initClass;
        PluginClass* pluginClass;


        MappingDesc* mappingDesc;

    public:
        SimulationStarter() : mappingDesc(nullptr)
        {
            simulationClass = new SimulationClass();
            initClass = new InitClass();
            simulationClass->setInitController(initClass);
            pluginClass = new PluginClass();
        }

        virtual ~SimulationStarter()
        {
            __delete(initClass);
            __delete(pluginClass);
            __delete(simulationClass);
        }

        virtual std::string pluginGetName() const
        {
            return "PIConGPU simulation starter";
        }

        virtual void start()
        {
            PluginConnector& pluginConnector = Environment<>::get().PluginConnector();
            pluginConnector.loadPlugins();
            log<picLog::SIMULATION_STATE>("Startup");
            simulationClass->setInitController(initClass);
            simulationClass->startSimulation();
        }

        virtual void pluginRegisterHelp(po::options_description&)
        {
        }

        void notify(uint32_t)
        {
        }

        ArgsParser::Status parseConfigs(int argc, char** argv)
        {
            ArgsParser& ap = ArgsParser::getInstance();
            PluginConnector& pluginConnector = Environment<>::get().PluginConnector();

            po::options_description simDesc(simulationClass->pluginGetName());
            simulationClass->pluginRegisterHelp(simDesc);
            ap.addOptions(simDesc);

            po::options_description initDesc(initClass->pluginGetName());
            initClass->pluginRegisterHelp(initDesc);
            ap.addOptions(initDesc);

            po::options_description pluginDesc(pluginClass->pluginGetName());
            pluginClass->pluginRegisterHelp(pluginDesc);
            ap.addOptions(pluginDesc);

            // setup all boost::program_options and add to ArgsParser
            BoostOptionsList options = pluginConnector.registerHelp();

            for(BoostOptionsList::const_iterator iter = options.begin(); iter != options.end(); ++iter)
            {
                ap.addOptions(*iter);
            }

            // parse environment variables, config files and command line
            return ap.parse(argc, argv);
        }

    protected:
        void pluginLoad()
        {
            simulationClass->load();
            mappingDesc = simulationClass->getMappingDescription();
            pluginClass->setMappingDescription(mappingDesc);
            initClass->setMappingDescription(mappingDesc);
        }

        void pluginUnload()
        {
            PluginConnector& pluginConnector = Environment<>::get().PluginConnector();
            pluginConnector.unloadPlugins();
            initClass->unload();
            pluginClass->unload();
            simulationClass->unload();
        }

    private:
        void printStartParameters(int argc, char** argv)
        {
            std::cout << "Start Parameters: ";
            for(int i = 0; i < argc; ++i)
            {
                std::cout << argv[i] << " ";
            }
            std::cout << std::endl;
        }
    };
} // namespace picongpu
