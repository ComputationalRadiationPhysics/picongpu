/**
 * Copyright 2013-2016 Rene Widera, Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc_types.hpp"
#include "identifier/value_identifier.hpp"
#include "identifier/alias.hpp"
#include "particles/frame_types.hpp"

namespace PMacc
{

/** position of a particle inside a supercell
 *
 * Value is a linear index inside the supercell
 */
value_identifier(lcellId_t,localCellIdx,0);

/** Is a value to set stages (is particle, is no particle,... of a particle
 *
 * if multiMask is set to:
 *  - 0 (zero) it is no particle
 *  - 1 it is a particle
 *  - 2 to 27 is used to define whether a particle leaf a supercell
 *    ExchangeType = value - 1 (e.g. 27 - 1 = 26 means particle leaves supercell
 *    over FRONT(value=18) TOP(value=6) LEFT(value=2) corner -> 18+6+2=26
 */
value_identifier(uint8_t,multiMask,0);

} //namespace PMacc
