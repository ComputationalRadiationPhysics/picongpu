/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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
 


#ifndef LASERPLANEWAVE_HPP
#define	LASERPLANEWAVE_HPP

#include "types.h"
#include "simulation_defines.hpp"

namespace picongpu
{
    /** plane wave (use periodic boundaries!)
     *
     *  no phase shifts, no spacial envelope
     */
    namespace laserPlaneWave
    {

        /** Compute the
         *
         */
        HINLINE float3_X laserLongitudinal( uint32_t currentStep, float_X& phase )
        {
            const double runTime = DELTA_T*currentStep;
            const double f = SPEED_OF_LIGHT / WAVE_LENGTH;

            float3_X elong = float3_X(float_X(0.0), float_X(0.0), float_X(0.0));

            // a NON-symmetric (starting with phase=0) pulse will be initialized at position z=0 for
            // a time of RAMP_INIT * PULSE_LENGTH + LASER_NOFOCUS_CONSTANT = INIT_TIME.
            // we shift the complete pulse for the half of this time to start with
            // the front of the laser pulse.
            const double mue = 0.5 * RAMP_INIT * PULSE_LENGTH;

            const double w = 2.0 * PI * f;

            const double endUpramp = mue;
            const double startDownramp = mue + LASER_NOFOCUS_CONSTANT;


            if( runTime >= endUpramp && runTime <= startDownramp )
            {
                // plateau
                elong.x() = float_X(
                                 double(AMPLITUDE )
                                 * sin( w * runTime )
                                 );
            }
            else if( runTime > startDownramp )
            {
                // downramp = end
                const double exponent =
                    ( ( runTime - startDownramp )
                      / PULSE_LENGTH / sqrt( 2.0 ) );
                elong.x() = float_X(
                                 double(AMPLITUDE )
                                 * exp( -0.5 * exponent * exponent )
                                 * sin( w * runTime )
                                 );
            }
            else
            {
                // upramp = start
                const double exponent = ( ( runTime - endUpramp ) / PULSE_LENGTH / sqrt( 2.0 ) );
                elong.x() = float_X(
                                 double(AMPLITUDE )
                                 * exp( -0.5 * exponent * exponent )
                                 * sin( w * runTime )
                                 );
            }

            phase = float_X(0.0);

            return elong;
        }
        
        /**
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

#endif	/* LASERPLANEWAVE_HPP */



