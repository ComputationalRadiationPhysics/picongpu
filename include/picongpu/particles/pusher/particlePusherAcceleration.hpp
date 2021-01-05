/* Copyright 2013-2021 Heiko Burau, Rene Widera,
 *                     Richard Pausch, Klaus Steiniger
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
    namespace particlePusherAcceleration
    {
        struct UnitlessParam : public particlePusherAccelerationParam
        {
            /** Normalize input values from `pusher.param` to PIC units */
            static constexpr float_X AMPLITUDEx = float_X(AMPLITUDEx_SI / UNIT_EFIELD); // unit: Volt / meter
            static constexpr float_X AMPLITUDEy = float_X(AMPLITUDEy_SI / UNIT_EFIELD); // unit: Volt / meter
            static constexpr float_X AMPLITUDEz = float_X(AMPLITUDEz_SI / UNIT_EFIELD); // unit: Volt / meter

            static constexpr float_X ACCELERATION_TIME = float_X(ACCELERATION_TIME_SI / UNIT_TIME); // unit: second
        };

        template<class Velocity, class Gamma>
        struct Push
        {
            /* this is an optional extension for sub-sampling pushes that enables grid to particle interpolation
             * for particle positions outside the super cell in one push
             */
            using LowerMargin = pmacc::math::CT::make_Int<simDim, 0>::type;
            using UpperMargin = pmacc::math::CT::make_Int<simDim, 0>::type;

            template<typename T_FunctorFieldE, typename T_FunctorFieldB, typename T_Particle, typename T_Pos>
            HDINLINE void operator()(
                const T_FunctorFieldB,
                const T_FunctorFieldE,
                T_Particle& particle,
                T_Pos& pos,
                const uint32_t currentStep)
            {
                using UnitlessParam = ::picongpu::particlePusherAcceleration::UnitlessParam;

                float_X const weighting = particle[weighting_];
                float_X const mass = attribute::getMass(weighting, particle);
                float_X const charge = attribute::getCharge(weighting, particle);

                using MomType = momentum::type;
                MomType new_mom = particle[momentum_];

                const float_X deltaT = DELTA_T;

                // normalize input SI values to
                const float3_X eField(UnitlessParam::AMPLITUDEx, UnitlessParam::AMPLITUDEy, UnitlessParam::AMPLITUDEz);

                /* ToDo: Refactor to ensure a smooth and slow increase of eField with time
                 * which may help to reduce radiation due to acceleration, if present.
                 */
                if(currentStep * DELTA_T <= UnitlessParam::ACCELERATION_TIME)
                    new_mom += charge * eField * deltaT;

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
                pmacc::traits::StringProperty propList("name", "Acceleration");
                return propList;
            }
        };
    } // namespace particlePusherAcceleration
} // namespace picongpu
