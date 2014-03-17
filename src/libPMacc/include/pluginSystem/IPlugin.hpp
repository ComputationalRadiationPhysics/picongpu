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

#include <stdexcept>
#include <string>


#include <boost/program_options/options_description.hpp>

namespace PMacc
{
    namespace po = boost::program_options;

    class PluginException : public std::runtime_error
    {
    public:

        PluginException(const char* message) : std::runtime_error(message)
        {
        }

        PluginException(std::string message) : std::runtime_error(message.c_str())
        {
        }
    };

    class IPlugin
    {
    public:

        IPlugin() :
        loaded(false)
        {

        }

        virtual ~IPlugin()
        {
        }

        virtual void load()
        {
            pluginLoad();
            loaded = true;
        }

        virtual void unload()
        {
            pluginUnload();
            loaded = false;
        }

        bool isLoaded()
        {
            return loaded;
        }

        virtual void pluginRegisterHelp(po::options_description& desc) = 0;

        virtual std::string pluginGetName() const = 0;

    protected:
        virtual void pluginLoad() = 0;

        virtual void pluginUnload() = 0;

        bool loaded;
    };
}
