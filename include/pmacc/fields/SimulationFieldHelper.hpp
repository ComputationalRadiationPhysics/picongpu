/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dimensions/GridLayout.hpp"
#include "pmacc/mappings/kernel/MappingDescription.hpp"

#include <cstdint>

namespace pmacc
{
    template<class CellDescription>
    class SimulationFieldHelper
    {
    public:
        using MappingDesc = CellDescription;

        static constexpr uint32_t dim = MappingDesc::Dim;

        SimulationFieldHelper(CellDescription description) : cellDescription(description)
        {
        }

        virtual ~SimulationFieldHelper() = default;

        /**
         * Reset is as well used for init.
         */
        virtual void reset(uint32_t currentStep) = 0;

        /**
         * Synchronize data from host to device.
         */
        virtual void syncToDevice() = 0;

        CellDescription getCellDescription() const
        {
            return cellDescription;
        }

    protected:
        CellDescription cellDescription;
    };

} // namespace pmacc
