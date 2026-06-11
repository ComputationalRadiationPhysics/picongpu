/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/boundary/Kind.hpp"

#include <cstdint>

namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            //! Description of a particle boundary
            struct Description
            {
                //! Boundary kind
                Kind kind = Kind::Absorbing;

                /** Offset inwards from the global domain boundary, in cells
                 *
                 * Is always non-negative and within the size of all local domains.
                 * Some boundary kinds may only support certain values of the offset.
                 */
                uint32_t offset = 0u;

                /** Boundary temperature in keV
                 *
                 * Only has effect for thermal boundaries
                 */
                float_X temperature = 0.0_X;
            };

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
