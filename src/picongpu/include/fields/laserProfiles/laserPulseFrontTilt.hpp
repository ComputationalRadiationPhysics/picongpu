/**
 * Copyright 2013 Heiko Burau, Anton Helm, Rene Widera
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



#ifndef LASERPULSEFRONTTILT
#define LASERPULSEFRONTTILT

#include "types.h"
#include "simulation_defines.hpp"

namespace picongpu
{
    namespace laserPulseFrontTilt
    {

        /**
         *
         * @param currentStep
         * @param subGrid
         * @param phase
         * @return
         */
        HINLINE float3_X laserLongitudinal( uint32_t currentStep, float_X& phase )
        {
            const double runTime = DELTA_T*currentStep;
            const double f = SPEED_OF_LIGHT / WAVE_LENGTH;

            float3_X elong = float3_X(float_X(0.0), float_X(0.0), float_X(0.0));

            // a symmetric pulse will be initialized at position z=0 for
            // a time of PULSE_INIT * PULSE_LENGTH = INIT_TIME.
            // we shift the complete pulse for the half of this time to start with
            // the front of the laser pulse.
            const double mue = 0.5 * INIT_TIME;

            //rayleigh length (in y-direction)
            const double y_R = PI * W0 * W0 / WAVE_LENGTH;
            //gaussian beam waist in the nearfield: w_y(y=0) == W0
            const double w_y = W0 * sqrt( 1.0 + ( FOCUS_POS / y_R )*( FOCUS_POS / y_R ) );


            const double envelope = double( AMPLITUDE ) * double( W0 ) / w_y;


            if( Polarisation == LINEAR_X )
            {
                elong.x() = float_X( envelope );
            }
            else if( Polarisation == LINEAR_Z )
            {
                elong.z() = float_X( envelope );
            }
            else if( Polarisation == CIRCULAR )
            {
                elong.x() = float_X( envelope / sqrt(2.0) );
                elong.z() = float_X( envelope / sqrt(2.0) );
            }

            phase = 2.0f * float_X(PI ) * float_X(f ) * ( runTime - float_X(mue ) - FOCUS_POS / SPEED_OF_LIGHT );

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
        HDINLINE float3_X laserTransversal( float3_X elong, float_X phase, const float_X posX, const float_X posZ )
        {
            //const float_X modMue = float_X(PI) * float_X(SPEED_OF_LIGHT / WAVE_LENGTH) * INIT_TIME;
            const float_X f = SPEED_OF_LIGHT / WAVE_LENGTH;
            const float_X timeShift = phase / (2.0f * float_X(PI) * float_X(f)) + FOCUS_POS / SPEED_OF_LIGHT;
            const float_X spaceShift = SPEED_OF_LIGHT * tanf(TILT_X) * timeShift / CELL_HEIGHT;
            const float_X r2 = (posX + spaceShift) * (posX + spaceShift) + posZ * posZ;

            // pure gaussian
            //const float_X r2 = posX * posX + posZ * posZ;

            //rayleigh length (in y-direction)
            const float_X y_R = float_X( PI ) * W0 * W0 / WAVE_LENGTH;

            // the radius of curvature of the beam's  wavefronts
            const float_X R_y = -FOCUS_POS * ( float_X(1.0) + ( y_R / FOCUS_POS )*( y_R / FOCUS_POS ) );


#if !defined(__CUDA_ARCH__) // Host code path
            //beam waist in the near field: w_y(y=0) == W0
            const float_X w_y = W0 * sqrt( float_X(1.0) + ( FOCUS_POS / y_R )*( FOCUS_POS / y_R ) );
            //! the Gouy phase shift
            const float_X xi_y = atan( -FOCUS_POS / y_R );

            if( Polarisation == LINEAR_X || Polarisation == LINEAR_Z )
            {
                elong *= math::exp( -r2 / w_y / w_y ) * cos( 2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + xi_y + phase )
                    * math::exp( -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                       *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                       / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH ) );
            }
            else if( Polarisation == CIRCULAR )
            {
                elong.x() *= math::exp( -r2 / w_y / w_y ) * cos( 2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + xi_y + phase )
                    * math::exp( -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                       *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                       / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH ) );
                phase += float_X( PI / 2.0 );
                elong.z() *= math::exp( -r2 / w_y / w_y ) * cos( 2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + xi_y + phase )
                    * math::exp( -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                       *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                       / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH ) );
                phase -= float_X( PI / 2.0 );
            }

            return elong;
#else
            //beam waist in the near field: w_y(y=0) == W0
            const float_X w_y = W0 * sqrtf( float_X(1.0) + ( FOCUS_POS / y_R )*( FOCUS_POS / y_R ) );
            //! the Gouy phase shift
            const float_X xi_y = atanf( -FOCUS_POS / y_R );

            if( Polarisation == LINEAR_X || Polarisation == LINEAR_Z )
            {
                elong *= math::exp( -r2 / w_y / w_y ) * __cosf( 2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + xi_y + phase )
                    * math::exp( -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                          *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                          / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH ) );
            }
            else if( Polarisation == CIRCULAR )
            {
                elong.x() *= math::exp( -r2 / w_y / w_y ) * __cosf( 2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + xi_y + phase )
                    * math::exp( -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                          *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                          / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH ) );
                phase += float_X( PI / 2.0 );
                elong.z() *= math::exp( -r2 / w_y / w_y ) * __cosf( 2.0f * float_X( PI ) / WAVE_LENGTH * FOCUS_POS - 2.0f * float_X( PI ) / WAVE_LENGTH * r2 / 2.0f / R_y + xi_y + phase )
                    * math::exp( -( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                          *( r2 / 2.0f / R_y - FOCUS_POS - phase / 2.0f / float_X( PI ) * WAVE_LENGTH )
                          / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0f * PULSE_LENGTH ) / ( 2.0f * PULSE_LENGTH ) );
                phase -= float_X( PI / 2.0 );
            }

            return elong;
#endif
        }

    }
}

#endif	/* LASERPULSEFRONTTILT_HPP */
