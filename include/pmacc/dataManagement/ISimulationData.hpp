/* Copyright 2013-2021 Rene Widera, Felix Schmitt, Benjamin Worpitz,
 *                     Alexander Grund
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <string>

namespace pmacc
{
    typedef std::string SimulationDataId;

    /**
     * Interface for simulation data which should be registered at DataConnector
     * for file output, visualization, etc.
     */
    class ISimulationData
    {
    public:
        virtual ~ISimulationData()
        {
        }
        /**
         * Synchronizes simulation data, meaning accessing (host side) data
         * will return up-to-date values.
         */
        virtual void synchronize() = 0;

        /**
         * Return the globally unique identifier for this simulation data.
         *
         * @return globally unique identifier
         */
        virtual SimulationDataId getUniqueId() = 0;
    };
} // namespace pmacc
