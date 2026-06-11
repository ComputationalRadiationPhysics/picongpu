/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera, Richard Pausch, Alexander Debus, Marco Garten, Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/eventSystem/Manager.hpp>

#include <cstdint>

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            //! Functor for the stage of the PIC loop performing particle push
            struct ParticlePush
            {
                /** Push all particle species
                 *
                 * @param step index of time iteration
                 * @param[out] commEvent particle communication event
                 */
                void operator()(uint32_t const step, pmacc::EventTask& commEvent) const;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
