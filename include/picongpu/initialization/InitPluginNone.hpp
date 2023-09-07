/* Copyright 2013-2023 Rene Widera
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
        void slide(uint32_t currentStep) override
        {
        }

        void init() override
        {
        }

        void printInformation() override
        {
        }

        void notify(uint32_t) override
        {
        }

        virtual ~InitPluginNone()
        {
        }

        void pluginRegisterHelp(po::options_description& desc) override
        {
        }

        std::string pluginGetName() const override
        {
            return "InitPluginNone";
        }

        void setMappingDescription(MappingDesc* cellDescription) override
        {
        }

    protected:
        void pluginLoad() override
        {
        }

        void pluginUnload() override
        {
        }
    };

} // namespace picongpu
