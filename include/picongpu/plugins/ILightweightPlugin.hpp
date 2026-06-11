/*
 * SPDX-FileCopyrightText: Felix Schmitt
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
        void restart(uint32_t, std::string const) override
        {
            // disable checkpoint/restart capabilities for lightweight plugins
        }

        void checkpoint(uint32_t, std::string const) override
        {
            // disable checkpoint/restart capabilities for lightweight plugins
        }

        ~ILightweightPlugin() override = default;
    };
} // namespace picongpu
