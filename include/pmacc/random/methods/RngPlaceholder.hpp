/*
 * SPDX-FileCopyrightText: Alexander Grund, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace random
    {
        namespace methods
        {
            //! placeholder for the rng method
            struct RngPlaceholder
            {
                using StateType = int;
            };

        } // namespace methods
    } // namespace random
} // namespace pmacc
