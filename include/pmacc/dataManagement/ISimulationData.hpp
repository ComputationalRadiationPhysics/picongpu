/*
 * SPDX-FileCopyrightText: Rene Widera, Felix Schmitt, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <string>

namespace pmacc
{
    using SimulationDataId = std::string;

    /**
     * Interface for simulation data which should be registered at DataConnector
     * for file output, visualization, etc.
     */
    class ISimulationData
    {
    public:
        virtual ~ISimulationData() = default;
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
