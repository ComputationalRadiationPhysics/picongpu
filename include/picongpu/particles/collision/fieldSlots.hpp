/*
 * SPDX-FileCopyrightText: Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once
#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            //! FieldTmp slot which is used to store the calculated Debye length
            constexpr uint32_t screeningLengthSlot = 0u;
        } // namespace collision
    } // namespace particles
} // namespace picongpu
