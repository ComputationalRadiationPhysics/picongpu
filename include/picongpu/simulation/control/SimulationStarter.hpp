/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
