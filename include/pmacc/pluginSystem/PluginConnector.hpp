/*
 * SPDX-FileCopyrightText: Rene Widera, Felix Schmitt, Axel Huebl, Benjamin Worpitz, Heiko Burau
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.def"
#include "pmacc/pluginSystem/INotify.hpp"
#include "pmacc/pluginSystem/IPlugin.hpp"
#include "pmacc/pluginSystem/Slice.hpp"

#include <list>
#include <string>
#include <vector>

namespace pmacc
{
    namespace po = boost::program_options;

    /**
     * Plugin registration and management class.
     */
    class PluginConnector
    {
    private:
        using SeqOfTimeSlices = std::vector<pluginSystem::Slice>;
        using PluginPair = std::pair<INotify*, SeqOfTimeSlices>;
        using NotificationList = std::list<PluginPair>;

    public:
        /** Register a plugin for loading/unloading and notifications
         *
         * Plugins are loaded in the order they are registered and unloaded in reverse order.
         * To trigger plugin notifications, call @see setNotificationPeriod after
         * registration.
         *
         * @param plugin plugin to register
         */
        void registerPlugin(IPlugin* plugin);

        /**
         * Calls load on all registered, not loaded plugins
         */
        void loadPlugins();

        /**
         * Unloads all registered, loaded plugins
         */
        void unloadPlugins();

        /**
         * Publishes command line parameters for registered plugins.
         *
         * @return list of boost program_options command line parameters
         */
        std::list<po::options_description> registerHelp();

        /** Set the notification period
         *
         * @param notifiedObj the object to notify, e.g. an IPlugin instance
         * @param period notification period
         */
        void setNotificationPeriod(INotify* notifiedObj, std::string const& period);

        /**
         * Notifies plugins that data should be dumped.
         *
         * @param currentStep current simulation iteration step
         */
        void notifyPlugins(uint32_t currentStep);

        /**
         * Notifies plugins that a restartable checkpoint should be dumped.
         *
         * @param currentStep current simulation iteration step
         * @param checkpointDirectory common directory for checkpoints
         */
        void checkpointPlugins(uint32_t currentStep, std::string const checkpointDirectory);

        /**
         * Notifies plugins that a restart is required.
         *
         * @param restartStep simulation iteration to restart from
         * @param restartDirectory common restart directory (contains checkpoints)
         */
        void restartPlugins(uint32_t restartStep, std::string const restartDirectory);

        /**
         * Get a vector of pointers of all registered plugin instances of a given type.
         *
         * @tparam Plugin type of plugin
         * @return vector of plugin pointers
         */
        template<typename Plugin>
        std::vector<Plugin*> getPluginsFromType()
        {
            {
                std::vector<Plugin*> result;
                for(auto iter = plugins.begin(); iter != plugins.end(); iter++)
                {
                    auto* plugin = dynamic_cast<Plugin*>(*iter);
                    if(plugin != nullptr)
                        result.push_back(plugin);
                }
                return result;
            }
        }

        /**
         * Return a copied list of pointers to all registered plugins.
         */
        std::list<IPlugin*> getAllPlugins() const;

    private:
        friend struct detail::Environment;

        static PluginConnector& getInstance()
        {
            static PluginConnector instance;
            return instance;
        }

        PluginConnector() = default;

        virtual ~PluginConnector() = default;

        std::list<IPlugin*> plugins;
        NotificationList notificationList;
    };
} // namespace pmacc
