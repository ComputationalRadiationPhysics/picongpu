/* Copyright 2014-2021 Felix Schmitt
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

#include "picongpu/plugins/ISimulationPlugin.hpp"

namespace picongpu
{
    /**
     * Interface for a lightweight simulation plugin
     * without checkpoint/restart capabilities.
     */
    class ILightweightPlugin : public ISimulationPlugin
    {
    public:
        void restart(uint32_t, const std::string)
        {
            // disable checkpoint/restart capabilities for lightweight plugins
        }

        void checkpoint(uint32_t, const std::string)
        {
            // disable checkpoint/restart capabilities for lightweight plugins
        }

        virtual ~ILightweightPlugin()
        {
        }
    };
} // namespace picongpu
