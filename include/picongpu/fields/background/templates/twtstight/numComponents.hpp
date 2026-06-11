/*
 * SPDX-FileCopyrightText: Alexander Debus, Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include <cstdint>

namespace picongpu
{
    namespace templates
    {
        namespace twtstight
        {
            namespace detail
            {
                /** Number of field components used in the simulation. [Default: 3 for both 2D and 3D] */
                uint32_t const numComponents = 3;
            } /* namespace detail */
        } /* namespace twtstight*/
    } /* namespace templates */
} /* namespace picongpu */
