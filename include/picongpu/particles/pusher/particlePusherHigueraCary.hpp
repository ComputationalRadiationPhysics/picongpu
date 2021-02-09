/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch, Annegret Roeszler, Klaus Steiniger
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
    namespace particlePusherHigueraCary
    {
        /** Implementation of the Higuera-Cary pusher as presented in doi:10.1063/1.4979989.
         *
         * A correction is applied to the given formulas as documented by the WarpX team:
         * (https://github.com/ECP-WarpX/WarpX/issues/320).
         *
         * Note, while Higuera and Ripperda present the formulas for the quantity u = gamma * v,
         * PIConGPU uses the real momentum p = gamma * m * v = u * m for calculations.
         * Here, all auxiliary quantities are equal to those used in Ripperda's article.
         *
         * Further references:
         * [Higuera's article on arxiv](https://arxiv.org/abs/1701.05605)
         * [Riperda's comparison of relativistic particle integrators](https://doi.org/10.3847/1538-4365/aab114)
         *
         * @tparam Velocity functor to compute the velocity of a particle with momentum p and mass m
         * @tparam Gamma functor to compute the Lorentz factor (= Energy/mc^2) of a particle with momentum p and mass m
         */
        template<typename Velocity, typename Gamma>
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
                float_X const mass = attribute::getMass(weighting, particle);
                float_X const charge = attribute::getCharge(weighting, particle);

                using MomType = momentum::type;
                MomType const mom = particle[momentum_];

                auto bField = functorBField(pos);
                auto eField = functorEField(pos);

                float_X const deltaT = DELTA_T;


                Gamma gamma;

                /* Momentum update
                 * Notation is according to Ripperda's paper
                 */
                // First half electric field acceleration
                namespace sqrt_HC = sqrt_HigueraCary;

                sqrt_HC::float3_X const mom_minus
                    = precisionCast<sqrt_HC::float_X>(mom + float_X(0.5) * charge * eField * deltaT);

                // Auxiliary quantitites
                sqrt_HC::float_X const gamma_minus = gamma(mom_minus, mass);

                sqrt_HC::float3_X const tau
                    = precisionCast<sqrt_HC::float_X>(float_X(0.5) * bField * charge * deltaT / mass);

                sqrt_HC::float_X const sigma = pmacc::math::abs2(gamma_minus) - pmacc::math::abs2(tau);

                sqrt_HC::float_X const u_star
                    = pmacc::math::dot(mom_minus, tau) / precisionCast<sqrt_HC::float_X>(mass * SPEED_OF_LIGHT);

                sqrt_HC::float_X const gamma_plus = math::sqrt(
                    sqrt_HC::float_X(0.5)
                    * (sigma
                       + math::sqrt(
                           pmacc::math::abs2(sigma)
                           + sqrt_HC::float_X(4.0) * (pmacc::math::abs2(tau) + pmacc::math::abs2(u_star)))));

                sqrt_HC::float3_X const t_vector = tau / gamma_plus;

                sqrt_HC::float_X const s
                    = sqrt_HC::float_X(1.0) / (sqrt_HC::float_X(1.0) + pmacc::math::abs2(t_vector));

                // Rotation step
                sqrt_HC::float3_X const mom_plus = s
                    * (mom_minus + pmacc::math::dot(mom_minus, t_vector) * t_vector
                       + pmacc::math::cross(mom_minus, t_vector));

                // Second half electric field acceleration (Note correction mom_minus -> mom_plus here compared to
                // Ripperda)
                MomType const mom_diff1 = float_X(0.5) * charge * eField * deltaT;
                MomType const mom_diff2 = precisionCast<float_X>(pmacc::math::cross(mom_plus, t_vector));
                MomType const mom_diff = mom_diff1 + mom_diff2;

                MomType const new_mom = precisionCast<float_X>(mom_plus) + mom_diff;

                particle[momentum_] = new_mom;

                // Position update
                Velocity velocity;

                float3_X const vel = velocity(new_mom, mass);

                for(uint32_t d = 0; d < simDim; ++d)
                {
                    pos[d] += (vel[d] * deltaT) / cellSize[d];
                }
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "other:Higuera-Cary");
                return propList;
            }
        };

    } // namespace particlePusherHigueraCary
} // namespace picongpu
