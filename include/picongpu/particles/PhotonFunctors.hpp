/* Copyright 2021 Pawel Ordyna
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

#include <pmacc/math/Vector.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>
namespace picongpu
{
    namespace particles
    {
        template<typename T_Species>
        struct GetAngFrequency
        {
            using Species = T_Species;
            HDINLINE float_X operator()() const
            {
                using FrameType = typename Species::FrameType;
                using WavelengthFlag =
                    typename pmacc::traits::Resolve<typename GetFlagType<FrameType, wavelength<>>::type>::type;
                return pmacc::math::Pi<float_X>::doubleValue * SPEED_OF_LIGHT / WavelengthFlag::getValue();
            }

            template<typename T_Particle>
            HDINLINE float_X operator()(const T_Particle& particle) const
            {
                float_X weighting = particle[weighting_];
                float_X momentum = math::abs(particle[momentum_]);
                return momentum / HBAR / weighting * SPEED_OF_LIGHT;
            }
        };

        /**
         * Returns the phase for a given timestep
         */
        template<typename T_Species>
        struct GetPhaseByTimestep
        {
            using Species = T_Species;

            HDINLINE float_64 calcPhase(uint32_t const& currentStep, float_64 const& omega, float_64 phi_0) const
            {
                /* phase phi = phi_0 - omega * t;
                 * Note: This MUST be calculated in double precision as single precision is inexact after ~100
                 * timesteps Double precision is enough for about 10^10 timesteps More timesteps (in SP&DP) are
                 * possible, if the product is implemented as a summation with summands reduced to 2*PI */
                const float_64 phaseDiffPerTimestep = fmod(omega * DELTA_T, 2 * PI);
                // Reduce summands to range of 2*PI to avoid bit canceling
                float_64 dPhi = fmod(phaseDiffPerTimestep * static_cast<float_64>(currentStep), 2 * PI);
                phi_0 = fmod(phi_0, 2 * PI);
                float_64 result = phi_0 - dPhi;
                // Keep in range of [0,2*PI)
                if(result < 0)
                    result += 2 * PI;
                return result;
            }

            HDINLINE float_64 operator()(const uint32_t currentStep, float_64 phi_0=0.0) const
            {
                const float_64 omega = GetAngFrequency<Species>()();
                return calcPhase(currentStep, omega, phi_0);
            }

            HDINLINE float_64 operator()(const uint32_t currentStep, Species const& particle, float_64 phi_0=0.0) const
            {
                const float_64 omega = GetAngFrequency<Species>()(particle);
                return calcPhase(currentStep, omega, phi_0);


            }
        };
    } // namespace particles
} // namespace picongpu
