/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace traits
    {
        template<typename T_CellType, typename T_Field, uint32_t T_simDim = simDim>
        struct FieldPosition;

    } // namespace traits
} // namespace picongpu
