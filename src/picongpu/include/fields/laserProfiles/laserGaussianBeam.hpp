/**
 * Copyright 2013-2017 Axel Huebl, Heiko Burau, Anton Helm, Rene Widera,
 *                     Richard Pausch, Alexander Debus
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

#include "pmacc_types.hpp"
#include "simulation_defines.hpp"

namespace picongpu
{
    namespace laserGaussianBeam
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
            const float_64 runTime = DELTA_T*currentStep;
            const float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;

            float3_X elong(float3_X::create(0.0));

            // a symmetric pulse will be initialized at position z=0 for
            // a time of PULSE_INIT * PULSE_LENGTH = INIT_TIME.
            // we shift the complete pulse for the half of this time to start with
            // the front of the laser pulse.
            const float_64 mue = 0.5 * INIT_TIME;

            //rayleigh length (in y-direction)
            const float_64 y_R = PI * W0 * W0 / WAVE_LENGTH;
            //gaussian beam waist in the nearfield: w_y(y=0) == W0
            const float_64 w_y = W0 * sqrt( 1.0 + ( FOCUS_POS / y_R )*( FOCUS_POS / y_R ) );

            float_64 envelope = float_64( AMPLITUDE );
            if( simDim == DIM2 )
                envelope *= math::sqrt( float_64( W0 ) / w_y );
            else if( simDim == DIM3 )
                envelope *= float_64( W0 ) / w_y;
            /* no 1D representation/implementation */

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

            phase = float_X(2.0) * float_X(PI ) * float_X(f ) * ( runTime - float_X(mue ) - FOCUS_POS / SPEED_OF_LIGHT) + LASER_PHASE;

            return elong;
        }

        /**
         *  Simple iteration algorithm to implement Laguerre polynomials for GPUs.
         *  @param n order of the Laguerre polynomial
         *  @param x coordinate at which the polynomial is evaluated
         *  @return
         */
        HDINLINE float_X simpleLaguerre( const uint32_t n, const float_X x )
        {
            //Result for special case n == 0
            if (n == 0) return float_X(1.0);
            uint32_t currentN = 1;
            float_X laguerreNMinus1 = float_X(1.0);
            float_X laguerreN = float_X(1.0) - x;
            float_X laguerreNPlus1 = float_X(0.0);
            while (currentN < n)
            {
                //Core statement of the algorithm
                laguerreNPlus1 = ( ( float_X(2.0) * float_X(currentN) + float_X(1.0) - x) * laguerreN - float_X(currentN) * laguerreNMinus1 ) / float_X(currentN + 1);
                //Advance by one order
                laguerreNMinus1 = laguerreN;
                laguerreN = laguerreNPlus1;
                currentN++;
            }
            return laguerreN;
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
            const float_X r2 = posX * posX + posZ * posZ;

            //rayleigh length (in y-direction)
            const float_X y_R = float_X( PI ) * W0 * W0 / WAVE_LENGTH;

            // the radius of curvature of the beam's  wavefronts
            const float_X R_y = -FOCUS_POS * ( float_X(1.0) + ( y_R / FOCUS_POS )*( y_R / FOCUS_POS ) );

            // initialize temporary variables
            float_X etrans = float_X(0.0);
            float_X etrans_norm = float_X(0.0);
            PMACC_CASSERT_MSG(MODENUMBER_must_be_smaller_than_number_of_entries_in_LAGUERREMODES_vector, MODENUMBER<LAGUERREMODES_t::dim);
            for ( uint32_t m = 0 ; m <= MODENUMBER ; ++m )
                etrans_norm += LAGUERREMODES[m];

            //beam waist in the near field: w_y(y=0) == W0
            const float_X w_y = W0 * algorithms::math::sqrt( float_X(1.0) + ( FOCUS_POS / y_R )*( FOCUS_POS / y_R ) );
            //! the Gouy phase shift
            const float_X xi_y = algorithms::math::atan( -FOCUS_POS / y_R );

            if( Polarisation == LINEAR_X || Polarisation == LINEAR_Z )
            {
                for ( uint32_t m = 0 ; m <= MODENUMBER ; ++m )
                {
                    etrans += LAGUERREMODES[m] * simpleLaguerre( m, float_X(2.0) * r2 / w_y / w_y )
                        * math::exp( -r2 / w_y / w_y ) * math::cos( float_X(2.0) * float_X( PI ) / WAVE_LENGTH * FOCUS_POS - float_X(2.0) * float_X( PI ) / WAVE_LENGTH * r2 / float_X(2.0) / R_y + ( 2*m + 1 ) * xi_y + phase )
                        * math::exp( -( r2 / float_X(2.0) / R_y - FOCUS_POS - phase / float_X(2.0) / float_X( PI ) * WAVE_LENGTH )
                              *( r2 / float_X(2.0) / R_y - FOCUS_POS - phase / float_X(2.0) / float_X( PI ) * WAVE_LENGTH )
                              / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( float_X(2.0) * PULSE_LENGTH ) / ( float_X(2.0) * PULSE_LENGTH ) );
                }
                elong *= etrans / etrans_norm;
            }
            else if( Polarisation == CIRCULAR )
            {
                for ( uint32_t m = 0 ; m <= MODENUMBER ; ++m )
                {
                    etrans += LAGUERREMODES[m] * simpleLaguerre( m, float_X(2.0) * r2 / w_y / w_y )
                        * math::exp( -r2 / w_y / w_y ) * math::cos( float_X(2.0) * float_X( PI ) / WAVE_LENGTH * FOCUS_POS - float_X(2.0) * float_X( PI ) / WAVE_LENGTH * r2 / float_X(2.0) / R_y + ( 2*m + 1 ) * xi_y + phase )
                        * math::exp( -( r2 / float_X(2.0) / R_y - FOCUS_POS - phase / float_X(2.0) / float_X( PI ) * WAVE_LENGTH )
                              *( r2 / float_X(2.0) / R_y - FOCUS_POS - phase / float_X(2.0) / float_X( PI ) * WAVE_LENGTH )
                              / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( float_X(2.0) * PULSE_LENGTH ) / ( float_X(2.0) * PULSE_LENGTH ) );
                }
                elong.x() *= etrans / etrans_norm;
                phase += float_X( PI / 2.0 );
                etrans = float_X(0.0);
                for ( uint32_t m = 0 ; m <= MODENUMBER ; ++m )
                {
                    etrans += LAGUERREMODES[m] * simpleLaguerre( m, float_X(2.0) * r2 / w_y / w_y )
                        * math::exp( -r2 / w_y / w_y ) * math::cos( float_X(2.0) * float_X( PI ) / WAVE_LENGTH * FOCUS_POS - float_X(2.0) * float_X( PI ) / WAVE_LENGTH * r2 / float_X(2.0) / R_y + ( 2*m + 1 ) * xi_y + phase )
                        * math::exp( -( r2 / float_X(2.0) / R_y - FOCUS_POS - phase / float_X(2.0) / float_X( PI ) * WAVE_LENGTH )
                              *( r2 / float_X(2.0) / R_y - FOCUS_POS - phase / float_X(2.0) / float_X( PI ) * WAVE_LENGTH )
                              / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( float_X(2.0) * PULSE_LENGTH ) / ( float_X(2.0) * PULSE_LENGTH ) );
                }
                elong.z() *= etrans / etrans_norm;
                phase -= float_X( PI / 2.0 );
            }

            return elong;

        }

    }
}

