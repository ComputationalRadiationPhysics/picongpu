/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/algorithms/Gamma.def"
#include "picongpu/defines.hpp"

namespace picongpu
{
    template<typename T_PrecisionType>
    template<typename T_MomType, typename T_MassType>
    HDINLINE T_PrecisionType Gamma<T_PrecisionType>::operator()(T_MomType const& mom, T_MassType const mass) const
    {
        valueType const fMom2 = pmacc::math::l2norm2(precisionCast<valueType>(mom));
        constexpr valueType c2 = sim.pic.getSpeedOfLight() * sim.pic.getSpeedOfLight();

        valueType const m2_c2_reci = valueType(1.0) / precisionCast<valueType>(mass * mass * c2);

        return math::sqrt(precisionCast<valueType>(valueType(1.0) + fMom2 * m2_c2_reci));
    }

} // namespace picongpu
