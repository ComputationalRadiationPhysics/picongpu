/* Copyright 2013-2019 Axel Huebl, Rene Widera, Felix Schmitt
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

#include "picongpu/simulation_defines.hpp"
#include <pmacc/pluginSystem/IPlugin.hpp>


namespace picongpu
{
    using namespace pmacc;

    /**
     * Interface for a simulation plugin in PIConGPU which has a MappingDesc.
     */
    class ISimulationPlugin : public IPlugin
    {
    public:
        virtual void setMappingDescription(MappingDesc *cellDescription) = 0;

        virtual ~ISimulationPlugin()
        {
        }
    };
}

