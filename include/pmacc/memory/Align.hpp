/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "pmacc/ppFunctions.hpp"

/** calculate and set the optimal alignment for data
 *
 * you must align all arrays and structs that are used on the device
 * @param byte size of data in bytes
 */
#define __optimal_align__(byte)                                                                                       \
    alignas(/** \bug avoid bug if alignment is >16 byte                                                               \
             * https://github.com/ComputationalRadiationPhysics/picongpu/issues/1563                                  \
             */                                                                                                       \
            PMACC_MIN(PMACC_ROUND_UP_NEXT_POW2(byte), 16))

#define PMACC_ALIGN(var, ...) __optimal_align__(sizeof(__VA_ARGS__)) __VA_ARGS__ var
#define PMACC_ALIGN8(var, ...) alignas(8) __VA_ARGS__ var
