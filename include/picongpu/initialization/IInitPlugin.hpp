/* Copyright 2013-2021 Rene Widera, Felix Schmitt
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

#include <pmacc/types.hpp>
#include <pmacc/pluginSystem/PluginConnector.hpp>
#include "picongpu/plugins/ILightweightPlugin.hpp"


namespace picongpu
{
    using namespace pmacc;

    class IInitPlugin : public ILightweightPlugin
    {
    public:
        virtual void slide(uint32_t currentStep) = 0;
        virtual void init() = 0;
        virtual void printInformation() = 0;

        virtual ~IInitPlugin()
        {
        }
    };
} // namespace picongpu
