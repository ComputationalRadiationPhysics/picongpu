/* Copyright 2017-2021 Axel Huebl, Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
#include "picongpu/algorithms/Gamma.hpp"


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
                return SPEED_OF_LIGHT * math::abs(precisionCast<ValueType>(mom));

            /* if mass is non-zero then gamma is well defined */
            const ValueType gamma = Gamma<ValueType>()(mom, mass);

            ValueType kinEnergy;

            if(gamma < GAMMA_THRESH)
            {
                const ValueType mom2 = pmacc::math::abs2(precisionCast<ValueType>(mom));
                /* non relativistic kinetic energy expression */
                kinEnergy = mom2 / (ValueType(2.0) * mass);
            }
            else
            {
                constexpr ValueType c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
                /* kinetic energy for particles: E = (gamma - 1) * m * c^2 */
                kinEnergy = (gamma - ValueType(1.0)) * mass * c2;
            }

            return kinEnergy;
        }
    };

} // namespace picongpu
