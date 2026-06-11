/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace densityProfiles
    {
        struct HomogenousImpl
        {
            template<typename T_SpeciesType>
            struct apply
            {
                using type = HomogenousImpl;
            };

            HINLINE HomogenousImpl(uint32_t currentStep)
            {
            }

            /** Calculate the normalized density
             *
             * @param totalCellOffset total offset including all slides [in cells]
             * @return float_X always 1.0
             */
            HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
            {
                return float_X(1.0);
            }
        };
    } // namespace densityProfiles
} // namespace picongpu
