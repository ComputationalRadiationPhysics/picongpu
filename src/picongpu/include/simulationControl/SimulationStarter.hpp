/**
 * Copyright 2013 Axel Huebl, René Widera
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
 


#ifndef SIMULATIONSTARTER_HPP
#define	SIMULATIONSTARTER_HPP

#include "types.h"
#include "simulation_defines.hpp"

#include <boost/program_options/options_description.hpp>
#include <cassert>
#include <iostream>

#include "simulation_defines.hpp"
#include "ArgsParser.hpp"
#include "dimensions/DataSpace.hpp"
#include "mappings/simulation/GridController.hpp"
#include "dimensions/GridLayout.hpp"
#include "mappings/kernel/MappingDescription.hpp"
#include "moduleSystem/ModuleConnector.hpp"
#include "moduleSystem/Module.hpp"
#include "simulationControl/ISimulationStarter.hpp"

namespace picongpu
{
    using namespace PMacc;

    template<class InitClass, class AnalyserClass, class SimulationClass>
    class SimulationStarter : public ISimulationStarter
    {
    private:
        typedef std::list<boost::program_options::options_description> BoostOptionsList;

        SimulationClass* simulationClass;
        InitClass* initClass;
        AnalyserClass* analyserClass;


        MappingDesc* mappingDesc;
    public:

        SimulationStarter() : mappingDesc(NULL)
        {
            simulationClass = new SimulationClass();
            initClass = new InitClass();
            simulationClass->setInitController(initClass);
            analyserClass = new AnalyserClass();
        }

        virtual ~SimulationStarter()
        {
            __delete(initClass);
            __delete(analyserClass);
            __delete(simulationClass);
        }

        virtual std::string moduleGetName() const
        {
            return "PIConGPU simulation starter";
        }

        virtual void start()
        {
            ModuleConnector& module_connector = ModuleConnector::getInstance();
            module_connector.loadModules();
            log<picLog::SIMULATION_STATE > ("Startup");
            simulationClass->setInitController(initClass);
            simulationClass->startSimulation();
        }

        virtual void moduleRegisterHelp(po::options_description&)
        {
        }

        bool parseConfigs(int argc, char **argv)
        {
            ArgsParser& ap = ArgsParser::getInstance();
            ModuleConnector& module_connector = ModuleConnector::getInstance();

            po::options_description simDesc(simulationClass->moduleGetName());
            simulationClass->moduleRegisterHelp(simDesc);
            ap.addOptions(simDesc);

            po::options_description initDesc(initClass->moduleGetName());
            initClass->moduleRegisterHelp(initDesc);
            ap.addOptions(initDesc);

            po::options_description analyserDesc(analyserClass->moduleGetName());
            analyserClass->moduleRegisterHelp(analyserDesc);
            ap.addOptions(analyserDesc);

            // setup all boost::program_options and add to ArgsParser
            BoostOptionsList options = module_connector.registerHelp();

            for (BoostOptionsList::const_iterator iter = options.begin();
                 iter != options.end(); ++iter)
            {
                ap.addOptions(*iter);
            }

            // parse environment variables, config files and command line
            return ap.parse(argc, argv);
        }
    protected:

        void moduleLoad()
        {
            simulationClass->load();
            mappingDesc = simulationClass->getMappingDescription();
            analyserClass->setMappingDescription(mappingDesc);
            initClass->setMappingDescription(mappingDesc);
        }

        void moduleUnload()
        {
            ModuleConnector& module_connector = ModuleConnector::getInstance();
            module_connector.unloadModules();
            initClass->unload();
            analyserClass->unload();
            simulationClass->unload();
        }
    private:

        void printStartParameters(int argc, char **argv)
        {
            std::cout << "Start Parameters: ";
            for (int i = 0; i < argc; ++i)
            {
                std::cout << argv[i] << " ";
            }
            std::cout << std::endl;
        }
    };
}

#endif	/* SIMULATIONSTARTER_HPP */

