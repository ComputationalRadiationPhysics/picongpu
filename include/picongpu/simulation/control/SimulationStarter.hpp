/* Copyright 2013-2024 Axel Huebl, Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/ArgsParser.hpp"
#include "picongpu/defines.hpp"
#include "picongpu/initialization/InitialiserController.hpp"
#include "picongpu/plugins/PluginController.hpp"
#include "picongpu/simulation/control/Simulation.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/pluginSystem/PluginConnector.hpp>

#include <boost/program_options/options_description.hpp>

#include <iostream>

namespace picongpu
{
    using namespace pmacc;

    class SimulationStarter : public IPlugin
    {
    private:
        using BoostOptionsList = std::list<boost::program_options::options_description>;
        Simulation simulationClass{};
        InitialiserController initClass{};
        PluginController pluginClass{};

        MappingDesc* mappingDesc{nullptr};

    public:
        SimulationStarter()
        {
            simulationClass.setInitController(initClass);
        }

        std::string pluginGetName() const override
        {
            return "PIConGPU simulation starter";
        }

        void start()
        {
            PluginConnector& pluginConnector = Environment<>::get().PluginConnector();
            pluginConnector.loadPlugins();
            log<picLog::SIMULATION_STATE>("Startup");
            simulationClass.startSimulation();
        }

        void pluginRegisterHelp(po::options_description&) override
        {
        }

        void notify(uint32_t) override
        {
        }

        ArgsParser::Status parseConfigs(int argc, char** argv)
        {
            ArgsParser& ap = ArgsParser::getInstance();
            PluginConnector& pluginConnector = Environment<>::get().PluginConnector();

            po::options_description simDesc(simulationClass.pluginGetName());
            simulationClass.pluginRegisterHelp(simDesc);
            ap.addOptions(simDesc);

            po::options_description initDesc(initClass.pluginGetName());
            initClass.pluginRegisterHelp(initDesc);
            ap.addOptions(initDesc);

            po::options_description pluginDesc(pluginClass.pluginGetName());
            pluginClass.pluginRegisterHelp(pluginDesc);
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

        void restart(uint32_t, std::string const) override
        {
            // nothing to do here
        }

        void checkpoint(uint32_t, std::string const) override
        {
            // nothing to do here
        }


    protected:
        void pluginLoad() override
        {
            simulationClass.load();
            mappingDesc = simulationClass.getMappingDescription();
            pluginClass.setMappingDescription(mappingDesc);
            initClass.setMappingDescription(mappingDesc);
        }

        void pluginUnload() override
        {
            PluginConnector& pluginConnector = Environment<>::get().PluginConnector();
            pluginConnector.unloadPlugins();
            initClass.unload();
            pluginClass.unload();
            simulationClass.unload();
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
