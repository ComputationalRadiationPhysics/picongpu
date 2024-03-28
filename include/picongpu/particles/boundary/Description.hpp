/* Copyright 2021-2023 Sergei Bastrakov
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

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
