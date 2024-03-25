/* Copyright 2013-2023 Rene Widera, Felix Schmitt, Axel Huebl, Benjamin Worpitz,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
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

    void PluginConnector::checkpointPlugins(uint32_t currentStep, const std::string checkpointDirectory)
    {
        for(auto iter = plugins.begin(); iter != plugins.end(); ++iter)
        {
            (*iter)->checkpoint(currentStep, checkpointDirectory);
            (*iter)->setLastCheckpoint(currentStep);
        }
    }

    void PluginConnector::restartPlugins(uint32_t restartStep, const std::string restartDirectory)
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
