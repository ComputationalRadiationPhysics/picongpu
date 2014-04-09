/**
 * Copyright 2013-2014 Rene Widera, Felix Schmitt
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
 
#pragma once

#include <list>

#include "pluginSystem/INotify.hpp"
#include "pluginSystem/IPlugin.hpp"

namespace PMacc
{
    namespace po = boost::program_options;

    /**
     * Plugin registration and management class.
     */
    class PluginConnector
    {
    public:

        /** Register a plugin for loading/unloading and notifications
         *
         * To trigger plugin notifications, call \see setNotificationFrequency after
         * registration.
         *
         * @param plugin plugin to register
         */
        void registerPlugin(IPlugin *plugin)
        throw (PluginException)
        {
            if (plugin != NULL)
            { 
                plugins.push_back(plugin);
            }
            else
                throw PluginException("Registering NULL as a plugin is not allowed.");
        }

        /**
         * Calls load on all registered, not loaded plugins
         */
        void loadPlugins()
        throw (PluginException)
        {
            // load all plugins
            for (std::list<IPlugin*>::reverse_iterator iter = plugins.rbegin();
                 iter != plugins.rend(); ++iter)
            {
                if (!(*iter)->isLoaded())
                {
                    (*iter)->load();
                }
            }
        }

        /**
         * Unloads all registered, loaded plugins
         */
        void unloadPlugins()
        throw (PluginException)
        {
            // unload all plugins
            for (std::list<IPlugin*>::reverse_iterator iter = plugins.rbegin();
                 iter != plugins.rend(); ++iter)
            {
                if ((*iter)->isLoaded())
                {
                    (*iter)->unload();
                }
            }
        }

        /**
         * Publishes command line parameters for registered plugins.
         *  
         * @return list of boost program_options command line parameters
         */
        std::list<po::options_description> registerHelp()
        {
            std::list<po::options_description> help_options;

            for (std::list<IPlugin*>::iterator iter = plugins.begin();
                 iter != plugins.end(); ++iter)
            {
                // create a new help options section for this plugin,
                // fill it and add to list of options
                po::options_description desc((*iter)->pluginGetName());
                (*iter)->pluginRegisterHelp(desc);
                help_options.push_back(desc);
            }

            return help_options;
        }

        /** Set the notification frequency
         *
         * Note: this works for a registered plugin but it is enought to implement
         *       the INotify interface.
         *
         * @param notify, e.g. an IPlugin to set a notify frequency for
         * @param frequency notification frequency
         */
        void setNotificationFrequency(INotify* notifiedObj, uint32_t frequency)
        {
            notificationMap[notifiedObj] = frequency;
        }

        /**
         * Notifies plugins that data should be dumped.
         *
         * @param currentStep current simulation iteration step
         */
        void notifyPlugins(uint32_t currentStep)
        {
            for (std::map<INotify*, uint32_t>::iterator iter = notificationMap.begin();
                    iter != notificationMap.end(); ++iter)
            {
                INotify* notifiedObj = iter->first;
                uint32_t frequency = iter->second;
                if (frequency > 0 && (currentStep % frequency == 0))
                    notifiedObj->notify(currentStep);
            }
        }

        /**
         * Notifies plugins that a restartable checkpoint should be dumped.
         * 
         * @param currentStep current simulation iteration step
         */
        void checkpointPlugins(uint32_t currentStep)
        {
            for (std::list<IPlugin*>::iterator iter = plugins.begin();
                    iter != plugins.end(); ++iter)
            {
                (*iter)->checkpoint(currentStep);
            }
        }
        
        /**
         * Notifies plugins that a restart is required.
         * 
         * @param restartStep simulation iteration to restart from
         */
        void restartPlugins(uint32_t restartStep)
        {
            for (std::list<IPlugin*>::iterator iter = plugins.begin();
                    iter != plugins.end(); ++iter)
            {
                (*iter)->restart(restartStep);
            }
        }

    private:
        
        friend Environment<DIM1>;
        friend Environment<DIM2>;
        friend Environment<DIM3>;
        
        static PluginConnector& getInstance()
        {
            static PluginConnector instance;
            return instance;
        }

        PluginConnector()
        {

        }

        virtual ~PluginConnector()
        {

        }

        std::list<IPlugin*> plugins;
        std::map<INotify*, uint32_t> notificationMap;
    };
}
