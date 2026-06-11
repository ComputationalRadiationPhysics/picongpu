/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/traits/attribute/GetCharge.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

#include <pmacc/traits/HasIdentifier.hpp>

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
                T_FunctorFieldB const functorBField,
                T_FunctorFieldE const functorEField,
                T_Particle& particle,
                T_Pos& pos,
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

                float_X const QoM = charge / mass;

                float_X const deltaT = sim.pic.getDt();

                MomType const mom_minus = mom + float_X(0.5) * charge * eField * deltaT;

                Gamma gamma;
                float_X const gamma_reci = float_X(1.0) / gamma(mom_minus, mass);
                float3_X const t = float_X(0.5) * QoM * bField * gamma_reci * deltaT;
                auto s = float_X(2.0) * t * (float_X(1.0) / (float_X(1.0) + pmacc::math::l2norm2(t)));

                MomType const mom_prime = mom_minus + pmacc::math::cross(mom_minus, t);
                MomType const mom_plus = mom_minus + pmacc::math::cross(mom_prime, s);

                MomType const new_mom = mom_plus + float_X(0.5) * charge * eField * deltaT;

                particle[momentum_] = new_mom;

                Velocity velocity;
                float3_X const vel = velocity(new_mom, mass);

                for(uint32_t d = 0; d < simDim; ++d)
                {
                    pos[d] += (vel[d] * deltaT) / sim.pic.getCellSize()[d];
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
