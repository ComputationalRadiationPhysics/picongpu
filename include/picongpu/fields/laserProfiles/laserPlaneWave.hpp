/* Copyright 2013-2018 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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

#include <pmacc/types.hpp>
#include "picongpu/simulation_defines.hpp"

namespace picongpu
{
    /** plane wave (use periodic boundaries!)
     *
     *  no transverse spacial envelope
     *  based on the electric potential
     *  Phi = Phi_0 * exp(0.5 * (x-x_0)^2 / sigma^2) * cos(k*(x - x_0) - phi)
     *  by applying -grad Phi = -d/dx Phi = E(x)
     *  we get:
     *  E = -Phi_0 * exp(0.5 * (x-x_0)^2 / sigma^2) * [k*sin(k*(x - x_0) - phi) + x/sigma^2 * cos(k*(x - x_0) - phi)]
     *
     *  This approach ensures that int_{-infinity}^{+infinity} E(x) = 0 for any phase
     *  if we have no transverse profile as we have with this plane wave train
     *
     *  Since PIConGPU requires a temporally defined electric field, we use:
     *  t = x/c and (x-x_0)/sigma = (t-t_0)/tau and k*(x-x_0) = omega*(t-t_0) with omega/k = c and tau * c = sigma
     *  and get:
     *  E = -Phi_0*omega/c * exp(0.5 * (t-t_0)^2 / tau^2) * [sin(omega*(t - t_0) - phi) + t/(omega*tau^2) * cos(omega*(t - t_0) - phi)]
     *  and define:
     *    E_0 = -Phi_0*omega/c
     *    integrationCorrectionFactor = t/(omega*tau^2)
     *
     *  Please consider:
     *   1) The above formulae does only apply to a Gaussian envelope. If the plateau length is
     *      not zero, the integral over the volume will only vanish if the plateau length is
     *      a multiple of the wavelength.
     *   2) Since we define our envelope by a sigma of the laser intensity,
     *      tau = PULSE_LENGTH * sqrt(2)
     */
    namespace laserPlaneWave
    {
        /** calculates longitudinal field distribution
         *
         * @param currentStep
         * @param phase
         * @return
         */
        HINLINE float3_X laserLongitudinal( uint32_t currentStep, float_X& phase )
        {
            /* initialize the laser not in the first cell is equal to a negative shift
             * in time
             */
            constexpr float_X laserTimeShift = laser::initPlaneY * CELL_HEIGHT / SPEED_OF_LIGHT;
            const float_64 runTime = DELTA_T * currentStep - laserTimeShift;
            const float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;

            float_64 envelope = float_64(AMPLITUDE );
            float3_X elong(float3_X::create(0.0));

            const float_64 mue = 0.5 * RAMP_INIT * PULSE_LENGTH;

            const float_64 w = 2.0 * PI * f;
            const float_64 tau = PULSE_LENGTH * sqrt( 2.0 );

            const float_64 endUpramp = mue;
            const float_64 startDownramp = mue + LASER_NOFOCUS_CONSTANT;

            float_64 integrationCorrectionFactor = 0.0;

            if( runTime > startDownramp )
            {
                // downramp = end
                const float_64 exponent = (runTime - startDownramp) / tau;
                envelope *= exp( -0.5 * exponent * exponent );
                integrationCorrectionFactor = ( runTime - startDownramp )/ (w*tau*tau);
            }
            else if ( runTime < endUpramp )
            {
                // upramp = start
                const float_64 exponent = (runTime - endUpramp) / tau;
                envelope *= exp( -0.5 * exponent * exponent );
                integrationCorrectionFactor = ( runTime - endUpramp )/ (w*tau*tau);
            }

            const float_64 timeOszi = runTime - endUpramp;
            const float_64 t_and_phase = w * timeOszi + LASER_PHASE;
            // to understand both components [sin(...) + t/tau^2 * cos(...)] see description above
            if( Polarisation == LINEAR_X )
            {
              elong.x() = float_X( envelope * (math::sin(t_and_phase)
                          + math::cos(t_and_phase) * integrationCorrectionFactor));
            }
            else if( Polarisation == LINEAR_Z)
            {
              elong.z() = float_X( envelope * (math::sin(t_and_phase)
                          + math::cos(t_and_phase) * integrationCorrectionFactor));
            }
            else if( Polarisation == CIRCULAR )
            {
                elong.x() = float_X( envelope / sqrt(2.0) * (math::sin(t_and_phase)
                            + math::cos(t_and_phase) * integrationCorrectionFactor));
                elong.z() = float_X( envelope / sqrt(2.0) * (math::cos(t_and_phase)
                            - math::sin(t_and_phase) * integrationCorrectionFactor));
            }


            phase = float_X(0.0);

            return elong;
        }

        /** calculates transverse field distribution
         *
         * @param elong
         * @param phase
         * @param posX
         * @param posZ
         * @return
         */
        HDINLINE float3_X laserTransversal( float3_X elong, const float_X, const float_X, const float_X )
        {
            return elong;
        }

    }
}

