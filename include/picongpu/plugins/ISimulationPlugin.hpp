/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera, Felix Schmitt
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "picongpu/defines.hpp"

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
        virtual void setMappingDescription(MappingDesc* cellDescription) = 0;

        ~ISimulationPlugin() override = default;
    };
} // namespace picongpu
