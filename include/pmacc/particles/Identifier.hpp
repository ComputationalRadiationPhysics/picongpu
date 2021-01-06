/* Copyright 2013-2021 Rene Widera, Alexander Grund, Axel Huebl
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/identifier/value_identifier.hpp"
#include "pmacc/identifier/alias.hpp"
#include "pmacc/particles/frame_types.hpp"


namespace pmacc
{
    /** cell of a particle inside a supercell
     *
     * Value is a linear cell index inside the supercell
     */
    value_identifier(lcellId_t, localCellIdx, 0);

    /** state of a particle
     *
     * Particle might be valid or invalid in a particle frame.
     * Valid particles can further be marked as candidates to leave a supercell.
     * Possible multiMask values are:
     *  - 0 (zero): no particle (invalid)
     *  - 1: particle (valid)
     *  - 2 to 27: (valid) particle that is about to leave its supercell but is
     *             still stored in the current particle frame.
     * Directions to leave the supercell are defined as follows.
     * An ExchangeType = value - 1 (e.g. 27 - 1 = 26) means particle leaves supercell
     * in the direction of FRONT(value=18) && TOP(value=6) && LEFT(value=2) which
     * defines a diagonal movement over a supercell corner (18+6+2=26).
     */
    value_identifier(uint8_t, multiMask, 0);

} // namespace pmacc
