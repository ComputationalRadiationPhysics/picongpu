/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

namespace pmacc
{
    namespace type
    {
        /*! area which is calculated
         *
         * CORE is the inner area of a grid
         * BORDER is the border of a grid (my own border, not the neighbor part)
         */
        enum AreaType
        {
            CORE = 1u,
            BORDER = 2u,
            GUARD = 4u
        };

    } // namespace type

    // for backward compatibility pull all definitions into the pmacc namespace
    using namespace type;
} // namespace pmacc
