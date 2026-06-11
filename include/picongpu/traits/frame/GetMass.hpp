/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace traits
    {
        namespace frame
        {
            template<typename T_Frame>
            HDINLINE float_X getMass();

        } // namespace frame
    } // namespace traits
} // namespace picongpu
