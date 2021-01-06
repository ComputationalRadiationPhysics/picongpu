/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch
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
#include "picongpu/traits/attribute/GetMass.hpp"
#include "picongpu/traits/attribute/GetCharge.hpp"


namespace picongpu
{
    namespace particlePusherBoris
    {
        template<class Velocity, class Gamma>
        struct Push
        {
            /* this is an optional extension for sub-sampling pushes that enables grid to particle interpolation
             * for particle positions outside the super cell in one push
             */
            using LowerMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;
            using UpperMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;

            template<typename T_FunctorFieldE, typename T_FunctorFieldB, typename T_Particle, typename T_Pos>
            HDINLINE void operator()(
                const T_FunctorFieldB functorBField,
                const T_FunctorFieldE functorEField,
                T_Particle& particle,
                T_Pos& pos,
                const uint32_t)
            {
                float_X const weighting = particle[weighting_];
                float_X const mass = attribute::getMass(weighting, particle);
                float_X const charge = attribute::getCharge(weighting, particle);

                using MomType = momentum::type;
                MomType const mom = particle[momentum_];

                auto bField = functorBField(pos);
                auto eField = functorEField(pos);

                const float_X QoM = charge / mass;

                const float_X deltaT = DELTA_T;

                const MomType mom_minus = mom + float_X(0.5) * charge * eField * deltaT;

                Gamma gamma;
                const float_X gamma_reci = float_X(1.0) / gamma(mom_minus, mass);
                const float3_X t = float_X(0.5) * QoM * bField * gamma_reci * deltaT;
                auto s = float_X(2.0) * t * (float_X(1.0) / (float_X(1.0) + pmacc::math::abs2(t)));

                const MomType mom_prime = mom_minus + pmacc::math::cross(mom_minus, t);
                const MomType mom_plus = mom_minus + pmacc::math::cross(mom_prime, s);

                const MomType new_mom = mom_plus + float_X(0.5) * charge * eField * deltaT;

                particle[momentum_] = new_mom;

                Velocity velocity;
                const float3_X vel = velocity(new_mom, mass);

                for(uint32_t d = 0; d < simDim; ++d)
                {
                    pos[d] += (vel[d] * deltaT) / cellSize[d];
                }
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "Boris");
                return propList;
            }
        };
    } // namespace particlePusherBoris
} // namespace picongpu
