/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/algorithms/Gamma.hpp"
#include "picongpu/defines.hpp"

namespace picongpu
{
    using namespace pmacc;

    /** Computes the kinetic energy of a particle given its momentum and mass.
     *
     * The mass may be zero.
     *
     * For massive particle with low energy the non-relativistic
     * kinetic energy expression is used in order to avoid bad roundings.
     *
     */
    template<typename T_PrecisionType = float_X>
    struct KinEnergy
    {
        using ValueType = T_PrecisionType;

        template<typename MomType, typename MassType>
        HDINLINE ValueType operator()(MomType const& mom, MassType const& mass)
        {
            if(mass == MassType(0.0))
                return sim.pic.getSpeedOfLight() * pmacc::math::l2norm(precisionCast<ValueType>(mom));

            /* if mass is non-zero then gamma is well defined */
            ValueType const gamma = Gamma<ValueType>()(mom, mass);

            ValueType kinEnergy;

            if(gamma < GAMMA_THRESH)
            {
                ValueType const mom2 = pmacc::math::l2norm2(precisionCast<ValueType>(mom));
                /* non relativistic kinetic energy expression */
                kinEnergy = mom2 / (ValueType(2.0) * mass);
            }
            else
            {
                constexpr ValueType c2 = sim.pic.getSpeedOfLight() * sim.pic.getSpeedOfLight();
                /* kinetic energy for particles: E = (gamma - 1) * m * c^2 */
                kinEnergy = (gamma - ValueType(1.0)) * mass * c2;
            }

            return kinEnergy;
        }
    };

} // namespace picongpu
