/*
 * SPDX-FileCopyrightText: Lennert Sprenger, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <cstdint>

namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            //! Basic parameters to be passed to some particle boundary condition functors
            struct Parameters
            {
                //! Axis of the active boundary
                uint32_t axis;

                /** Begin of the internal (relative to boundary) cells in total coordinates along the axis
                 *
                 * Particles with totalCellIdx[axis] < beginInternalCellsTotal are outside
                 */
                int32_t beginInternalCellsTotal;

                /** End of the internal (relative to boundary) cells in total coordinates along the axis
                 *
                 * Particles with totalCellIdx[axis] >= endInternalCellsTotal are outside
                 */
                int32_t endInternalCellsTotal;
            };
        } // namespace boundary
    } // namespace particles
} // namespace picongpu
