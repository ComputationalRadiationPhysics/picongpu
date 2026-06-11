/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
