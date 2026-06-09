/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/param/pusher.param"
#include "picongpu/traits/attribute/GetCharge.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

#include <pmacc/traits/HasIdentifier.hpp>

namespace picongpu
{
    namespace particlePusherVay
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
                T_FunctorFieldB const functorBField, /* at t=0 */
                T_FunctorFieldE const functorEField, /* at t=0 */
                T_Particle& particle,
                T_Pos& pos, /* at t=0 */
                uint32_t const)
            {
                float_X const weighting = particle[weighting_];
                float_X const mass = picongpu::traits::attribute::getMass(weighting, particle);
                float_X const charge = picongpu::traits::attribute::getCharge(weighting, particle);

                using MomType = momentum::type;
                MomType const mom = particle[momentum_];

                auto const bField = functorBField(pos);
                auto const eField = functorEField(pos);

                // update probe field if particle contains required attributes
                if constexpr(pmacc::traits::HasIdentifier<T_Particle, probeB>::type::value)
                    particle[probeB_] = bField;
                if constexpr(pmacc::traits::HasIdentifier<T_Particle, probeE>::type::value)
                    particle[probeE_] = eField;
                /*
                     time index in paper is reduced by a half: i=0 --> i=-1/2 so that momenta are
                     at half time steps and fields and locations are at full time steps

                     Here the real (PIConGPU) momentum (p) is used, not the momentum from the Vay paper (u)
                     p = m_0 * u
                */
                float_X const deltaT = sim.pic.getDt();
                float_X const factor = 0.5 * charge * deltaT;
                Gamma gamma;
                Velocity velocity;

                // first step in Vay paper:
                float3_X const velocity_atMinusHalf = velocity(mom, mass);
                // mom /(mass*mass + abs2(mom)/(sim.pic.getSpeedOfLight()*sim.pic.getSpeedOfLight()));
                MomType const momentum_atZero
                    = mom + factor * (eField + pmacc::math::cross(velocity_atMinusHalf, bField));

                // second step in Vay paper:
                MomType const momentum_prime = momentum_atZero + factor * eField;
                float_X const gamma_prime = gamma(momentum_prime, mass);

                sqrt_Vay::float3_X const tau(factor / mass * bField);
                sqrt_Vay::float_X const u_star
                    = pmacc::math::dot(precisionCast<sqrt_Vay::float_X>(momentum_prime), tau)
                      / precisionCast<sqrt_Vay::float_X>(sim.pic.getSpeedOfLight() * mass);
                sqrt_Vay::float_X const sigma = gamma_prime * gamma_prime - pmacc::math::l2norm2(tau);
                sqrt_Vay::float_X const gamma_atPlusHalf = math::sqrt(
                    sqrt_Vay::float_X(0.5)
                    * (sigma
                       + math::sqrt(
                           sigma * sigma + sqrt_Vay::float_X(4.0) * (pmacc::math::l2norm2(tau) + u_star * u_star))));
                float3_X const t(tau * (float_X(1.0) / gamma_atPlusHalf));
                float_X const s = float_X(1.0) / (float_X(1.0) + pmacc::math::l2norm2(t));
                MomType const momentum_atPlusHalf = s
                                                    * (momentum_prime + pmacc::math::dot(momentum_prime, t) * t
                                                       + pmacc::math::cross(momentum_prime, t));

                particle[momentum_] = momentum_atPlusHalf;

                float3_X const vel = velocity(momentum_atPlusHalf, mass);

                for(uint32_t d = 0; d < simDim; ++d)
                {
                    pos[d] += (vel[d] * sim.pic.getDt()) / sim.pic.getCellSize()[d];
                }
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "Vay");
                return propList;
            }
        };
    } // namespace particlePusherVay
} // namespace picongpu
