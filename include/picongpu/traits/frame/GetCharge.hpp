/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/traits/HasIdentifier.hpp>

namespace picongpu
{
    namespace traits
    {
        namespace frame
        {
            /** get the charge value for a species frame
             */
            template<typename T_Frame>
            HDINLINE float_X getCharge();

        } // namespace frame
    } // namespace traits
} // namespace picongpu
