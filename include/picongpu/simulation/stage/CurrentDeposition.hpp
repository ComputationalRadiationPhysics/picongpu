/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera, Richard Pausch, Alexander Debus, Marco
 * Garten, Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <cstdint>

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            //! Functor for the stage of the PIC loop performing current deposition
            struct CurrentDeposition
            {
                /** Compute the current created by particles and add it to the current
                 *  density
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
