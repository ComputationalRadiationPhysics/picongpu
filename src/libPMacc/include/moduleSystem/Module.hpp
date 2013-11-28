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
 

#ifndef MODULE_HPP
#define	MODULE_HPP

#include <stdexcept>
#include <string>


#include <boost/program_options/options_description.hpp>

namespace PMacc
{
    namespace po = boost::program_options;

    class ModuleException : /*virtual*/ public std::runtime_error
    {
    public:

        ModuleException(const char* message) : std::runtime_error(message)
        {
        }

        ModuleException(std::string message) : std::runtime_error(message.c_str())
        {
        }
    };

    class Module
    {
    public:

        Module() :
        loaded(false)
        {

        }

        virtual ~Module()
        {
        }

        virtual void load()
        {
            moduleLoad();
            loaded = true;
        }

        virtual void unload()
        {
            moduleUnload();
            loaded = false;
        }

        bool isLoaded()
        {
            return loaded;
        }

        virtual void moduleRegisterHelp(po::options_description& desc) = 0;

        virtual std::string moduleGetName() const = 0;

    protected:
        virtual void moduleLoad() = 0;

        virtual void moduleUnload() = 0;

        bool loaded;
    };
}

#endif	/* MODULE_HPP */

