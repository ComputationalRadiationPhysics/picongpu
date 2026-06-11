/*
 * SPDX-FileCopyrightText: Jakob Trojok
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** return type for ionization algorithms
             */
            struct IonizerReturn
            {
                float_X ionizationEnergy = 0._X;
                uint32_t newMacroElectrons = 0u;
            };
        } // namespace ionization
    } // namespace particles
} // namespace picongpu
