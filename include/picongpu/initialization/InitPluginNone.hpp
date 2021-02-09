/* Copyright 2013-2021 Rene Widera
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


#pragma once

#include "picongpu/initialization/IInitPlugin.hpp"


namespace picongpu
{
    using namespace pmacc;

    class InitPluginNone : public IInitPlugin
    {
    public:
        virtual void slide(uint32_t currentStep)
        {
        }

        virtual void init()
        {
        }

        virtual void printInformation()
        {
        }

        void notify(uint32_t)
        {
        }

        virtual ~InitPluginNone()
        {
        }

        virtual void pluginRegisterHelp(po::options_description& desc)
        {
        }

        virtual std::string pluginGetName() const
        {
            return "InitPluginNone";
        }

        virtual void setMappingDescription(MappingDesc* cellDescription)
        {
        }

    protected:
        virtual void pluginLoad()
        {
        }

        virtual void pluginUnload()
        {
        }
    };

} // namespace picongpu
