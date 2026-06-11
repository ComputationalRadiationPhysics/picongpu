/*
 * SPDX-FileCopyrightText: Rene Widera, Felix Schmitt, Axel Huebl, Benjamin Worpitz, Heiko Burau
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pmacc/pluginSystem/PluginConnector.hpp"

#include "pmacc/pluginSystem/INotify.hpp"
#include "pmacc/pluginSystem/IPlugin.hpp"
#include "pmacc/pluginSystem/Slice.hpp"
#include "pmacc/pluginSystem/containsStep.hpp"
#include "pmacc/pluginSystem/toSlice.hpp"

#include <list>
#include <string>
#include <vector>

namespace pmacc
{
    void PluginConnector::registerPlugin(IPlugin* plugin)
    {
        if(plugin != nullptr)
        {
            plugins.push_back(plugin);
        }
        else
            throw PluginException("Registering nullptr as a plugin is not allowed.");
    }

    void PluginConnector::loadPlugins()
    {
        // load all plugins
        for(auto iter = plugins.begin(); iter != plugins.end(); ++iter)
        {
            if(!(*iter)->isLoaded())
            {
                (*iter)->load();
            }
        }
    }

    void PluginConnector::unloadPlugins()
    {
        // unload all plugins
        for(auto iter = plugins.rbegin(); iter != plugins.rend(); ++iter)
        {
            if((*iter)->isLoaded())
            {
                (*iter)->unload();
            }
        }
        // Make sure plugin instances are deleted and so resources are freed
        plugins.clear();
    }

    std::list<po::options_description> PluginConnector::registerHelp()
    {
        std::list<po::options_description> help_options;

        for(auto iter = plugins.begin(); iter != plugins.end(); ++iter)
        {
            // create a new help options section for this plugin,
            // fill it and add to list of options
            po::options_description desc((*iter)->pluginGetName());
            (*iter)->pluginRegisterHelp(desc);
            help_options.push_back(desc);
        }

        return help_options;
    }

    void PluginConnector::setNotificationPeriod(INotify* notifiedObj, std::string const& period)
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

    void PluginConnector::notifyPlugins(uint32_t currentStep)
    {
        for(auto iter = notificationList.begin(); iter != notificationList.end(); ++iter)
        {
            if(containsStep((*iter).second, currentStep))
            {
                INotify* notifiedObj = iter->first;
                notifiedObj->notify(currentStep);
                notifiedObj->setLastNotify(currentStep);
            }
        }
    }

    void PluginConnector::checkpointPlugins(uint32_t currentStep, std::string const checkpointDirectory)
    {
        for(auto iter = plugins.begin(); iter != plugins.end(); ++iter)
        {
            (*iter)->checkpoint(currentStep, checkpointDirectory);
            (*iter)->setLastCheckpoint(currentStep);
        }
    }

    void PluginConnector::restartPlugins(uint32_t restartStep, std::string const restartDirectory)
    {
        for(auto iter = plugins.begin(); iter != plugins.end(); ++iter)
        {
            (*iter)->restart(restartStep, restartDirectory);
        }
    }

    std::list<IPlugin*> PluginConnector::getAllPlugins() const
    {
        return this->plugins;
    }

} // namespace pmacc
