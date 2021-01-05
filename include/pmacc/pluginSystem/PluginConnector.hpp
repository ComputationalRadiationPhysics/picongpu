/* Copyright 2013-2021 Rene Widera, Felix Schmitt, Axel Huebl, Benjamin Worpitz,
 *                     Heiko Burau
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

#include "pmacc/pluginSystem/INotify.hpp"
#include "pmacc/pluginSystem/IPlugin.hpp"
#include "pmacc/pluginSystem/TimeSlice.hpp"
#include "pmacc/pluginSystem/toTimeSlice.hpp"
#include "pmacc/pluginSystem/containsStep.hpp"

#include <vector>
#include <list>
#include <string>


namespace pmacc
{
    namespace po = boost::program_options;

    /**
     * Plugin registration and management class.
     */
    class PluginConnector
    {
    private:
        using SeqOfTimeSlices = std::vector<pluginSystem::TimeSlice>;
        using PluginPair = std::pair<INotify*, SeqOfTimeSlices>;
        using NotificationList = std::list<PluginPair>;

    public:
        /** Register a plugin for loading/unloading and notifications
         *
         * Plugins are loaded in the order they are registered and unloaded in reverse order.
         * To trigger plugin notifications, call \see setNotificationPeriod after
         * registration.
         *
         * @param plugin plugin to register
         */
        void registerPlugin(IPlugin* plugin)
        {
            if(plugin != nullptr)
            {
                plugins.push_back(plugin);
            }
            else
                throw PluginException("Registering nullptr as a plugin is not allowed.");
        }

        /**
         * Calls load on all registered, not loaded plugins
         */
        void loadPlugins()
        {
            // load all plugins
            for(std::list<IPlugin*>::iterator iter = plugins.begin(); iter != plugins.end(); ++iter)
            {
                if(!(*iter)->isLoaded())
                {
                    (*iter)->load();
                }
            }
        }

        /**
         * Unloads all registered, loaded plugins
         */
        void unloadPlugins()
        {
            // unload all plugins
            for(std::list<IPlugin*>::reverse_iterator iter = plugins.rbegin(); iter != plugins.rend(); ++iter)
            {
                if((*iter)->isLoaded())
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

            for(std::list<IPlugin*>::iterator iter = plugins.begin(); iter != plugins.end(); ++iter)
            {
                // create a new help options section for this plugin,
                // fill it and add to list of options
                po::options_description desc((*iter)->pluginGetName());
                (*iter)->pluginRegisterHelp(desc);
                help_options.push_back(desc);
            }

            return help_options;
        }

        /** Set the notification period
         *
         * @param notifiedObj the object to notify, e.g. an IPlugin instance
         * @param period notification period
         */
        void setNotificationPeriod(INotify* notifiedObj, std::string const& period)
        {
            if(notifiedObj != nullptr)
            {
                if(!period.empty())
                {
                    SeqOfTimeSlices seqTimeSlices = pluginSystem::toTimeSlice(period);
                    notificationList.push_back(std::make_pair(notifiedObj, seqTimeSlices));
                }
            }
            else
                throw PluginException("Notifications for a nullptr object are not allowed.");
        }

        /**
         * Notifies plugins that data should be dumped.
         *
         * @param currentStep current simulation iteration step
         */
        void notifyPlugins(uint32_t currentStep)
        {
            for(NotificationList::iterator iter = notificationList.begin(); iter != notificationList.end(); ++iter)
            {
                if(containsStep((*iter).second, currentStep))
                {
                    INotify* notifiedObj = iter->first;
                    notifiedObj->notify(currentStep);
                    notifiedObj->setLastNotify(currentStep);
                }
            }
        }

        /**
         * Notifies plugins that a restartable checkpoint should be dumped.
         *
         * @param currentStep current simulation iteration step
         * @param checkpointDirectory common directory for checkpoints
         */
        void checkpointPlugins(uint32_t currentStep, const std::string checkpointDirectory)
        {
            for(std::list<IPlugin*>::iterator iter = plugins.begin(); iter != plugins.end(); ++iter)
            {
                (*iter)->checkpoint(currentStep, checkpointDirectory);
                (*iter)->setLastCheckpoint(currentStep);
            }
        }

        /**
         * Notifies plugins that a restart is required.
         *
         * @param restartStep simulation iteration to restart from
         * @param restartDirectory common restart directory (contains checkpoints)
         */
        void restartPlugins(uint32_t restartStep, const std::string restartDirectory)
        {
            for(std::list<IPlugin*>::iterator iter = plugins.begin(); iter != plugins.end(); ++iter)
            {
                (*iter)->restart(restartStep, restartDirectory);
            }
        }

        /**
         * Get a vector of pointers of all registered plugin instances of a given type.
         *
         * \tparam Plugin type of plugin
         * @return vector of plugin pointers
         */
        template<typename Plugin>
        std::vector<Plugin*> getPluginsFromType()
        {
            std::vector<Plugin*> result;
            for(std::list<IPlugin*>::iterator iter = plugins.begin(); iter != plugins.end(); iter++)
            {
                Plugin* plugin = dynamic_cast<Plugin*>(*iter);
                if(plugin != nullptr)
                    result.push_back(plugin);
            }
            return result;
        }


        /**
         * Return a copied list of pointers to all registered plugins.
         */
        std::list<IPlugin*> getAllPlugins() const
        {
            return this->plugins;
        }

    private:
        friend struct detail::Environment;

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
        NotificationList notificationList;
    };
} // namespace pmacc
