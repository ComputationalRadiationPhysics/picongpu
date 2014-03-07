/**
 * Copyright 2013 Rene Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
 
#ifndef MODULECONNECTOR_HPP
#define	MODULECONNECTOR_HPP

#include <list>

#include "moduleSystem/Module.hpp"

namespace PMacc
{
    namespace po = boost::program_options;

    class ModuleConnector
    {
    public:

        void registerModule(Module *module)
        throw (ModuleException)
        {
            if (module != NULL)
            {
                modules.push_back(module);
            }
            else
                throw ModuleException("Registering NULL as a module is not allowed.");
        }

        void loadModules()
        throw (ModuleException)
        {
            // load all modules
            for (std::list<Module*>::reverse_iterator iter = modules.rbegin();
                 iter != modules.rend(); ++iter)
            {
                if (!(*iter)->isLoaded())
                {
                    (*iter)->load();
                }
            }
        }

        void unloadModules()
        throw (ModuleException)
        {
            // unload all modules
            for (std::list<Module*>::reverse_iterator iter = modules.rbegin();
                 iter != modules.rend(); ++iter)
            {
                if ((*iter)->isLoaded())
                {
                    (*iter)->unload();
                }
            }
        }

        std::list<po::options_description> registerHelp()
        {
            std::list<po::options_description> help_options;

            for (std::list<Module*>::iterator iter = modules.begin();
                 iter != modules.end(); ++iter)
            {
                // create a new help options section for this module,
                // fill it and add to list of options
                po::options_description desc((*iter)->moduleGetName());
                (*iter)->moduleRegisterHelp(desc);
                help_options.push_back(desc);
            }

            return help_options;
        }

    private:
        
        friend Environment<DIM1>;
        friend Environment<DIM2>;
        friend Environment<DIM3>;
        
        static ModuleConnector& getInstance()
        {
            static ModuleConnector instance;
            return instance;
        }

        ModuleConnector()
        {

        }

        virtual ~ModuleConnector()
        {

        }

        std::list<Module*> modules;
    };
}

#endif	/* MODULECONNECTOR_HPP */

