/*
 * SPDX-FileCopyrightText: Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

namespace picongpu
{
    namespace traits
    {
        template<typename T_Species>
        struct GetPusher
        {
            using type = typename pmacc::traits::Resolve<
                typename pmacc::traits::GetFlagType<typename T_Species::FrameType, particlePusher<>>::type>::type;
        };

    } // namespace traits
} // namespace picongpu
