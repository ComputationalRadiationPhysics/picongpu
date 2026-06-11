/*
 * SPDX-FileCopyrightText: Rene Widera
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
            //! Initialize particles
            struct ParticleInit
            {
                /** Initialize particles dependent of the given step
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const;
            };
        } // namespace stage
    } // namespace simulation
} // namespace picongpu
