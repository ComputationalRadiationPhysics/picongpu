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

        /**
         * Register a plugin for loading/unloading and notifications.
         * To allow plugin notifications, call setNotificationFrequency after registration.
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
        
        /**
         * Set the notification frequency for a registered plugin.
         * 
         * @param plugin plugin to set frequency for
         * @param frequency notification frequency
         */
        void setNotificationFrequency(IPlugin* plugin, uint32_t frequency)
        {
            notificationMap[plugin] = frequency;
        }
        
        /**
         * Notifies observers that data should be dumped.
         *
         * @param currentStep current simulation iteration step
         */
        void notifyPlugins(uint32_t currentStep)
        {
            for (std::list<IPlugin*>::iterator iter = plugins.begin();
                    iter != plugins.end(); ++iter)
            {
                IPlugin* plugin = *iter;
                if (notificationMap.find(plugin) != notificationMap.end())
                {
                    uint32_t frequency = notificationMap[plugin];
                    if (frequency > 0 && (currentStep % frequency == 0))
                        plugin->notify(currentStep);
                }
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
        std::map<IPlugin*, uint32_t> notificationMap;
    };
}
