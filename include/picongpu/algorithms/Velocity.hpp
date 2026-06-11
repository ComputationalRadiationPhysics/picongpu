/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    using namespace pmacc;

    struct Velocity
    {
        template<typename MomType, typename MassType>
        HDINLINE MomType operator()(MomType const mom, MassType const mass0)
        {
            float_X const rc2 = sim.pic.getMue0Eps0();
            float_X const m0_2 = mass0 * mass0;
            float_X const fMom2 = pmacc::math::l2norm2(mom);
            float_X t = math::rsqrt(precisionCast<sqrt_X>(m0_2 + fMom2 * rc2));
            return t * mom;
        }
    };
} // namespace picongpu
