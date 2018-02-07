/* Copyright 2013-2017 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Stefan Tietze
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


namespace picongpu
{
/** Wavepacket with spatial Gaussian envelope and adjustable temporal shape.
 * Allows defining a prepulse and two regions of exponential preramp with
 * independent slopes. The definition works by specifying three (t, intensity)-
 * points, where time is counted from the very beginning in SI and the
 * intensity (yes, intensity, not amplitude) is given in multiples of the main
 * peak.
 *
 * Be careful - problematic for few cycle pulses. Thought the rest is cloned
 * from laserWavepacket, the correctionFactor is not included (this made a
 * correction to the laser phase, which is necessary for very short pulses,
 * since otherwise a test particle is, after the laser pulse has passed, not
 * returned to immobility, as it should). Since the analytical solution is
 * only implemented for the Gaussian regime, and we have mostly exponential
 * regimes here, it was not retained here.
 */

namespace laserExpRampWithPrepulse
{
    constexpr float_X laserTimeShift = laser::initPlaneY * CELL_HEIGHT /
        SPEED_OF_LIGHT;
    constexpr float_X time_start_init = TIME_1 -
        ( 0.5 * RAMP_INIT * PULSE_LENGTH );
    constexpr float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;
    constexpr float_64 w = 2.0 * PI * f;

    /** takes time t relative to the center of the Gaussian and returns value
     * between 0 and 1, i.e. as multiple of the max value.
     * use as: amp_t = amp_0 * gauss( t - t_0 )
     */
    HDINLINE float_X
    gauss( float_X const t )
    {
        float_X const exponent = t / float_X( PULSE_LENGTH );
        return math::exp( float_X( -0.25 ) * exponent * exponent );
    }

    /** get value of exponential curve through two points at given t
     * t/t1/t2 given as float_X, since the envelope doesn't need the accuracy
     */
    HDINLINE float_X
    extrapolate_expo(
        float_X const t1,
        float_X const a1,
        float_X const t2,
        float_X const a2,
        float_X const t
    )
    {
        const float_X log1 = ( t2 - t ) * math::log( a1 );
        const float_X log2 = ( t - t1 ) * math::log( a2 );
        return math::exp( ( log1 + log2 )/( t2 - t1 ) );
    }

    HINLINE float_X
    get_envelope(float_X runTime)
    {
        float_X env = 0.0;
        const bool before_preupramp = runTime < time_start_init;
        const bool before_start = runTime < TIME_1;
        const bool before_peakpulse = runTime < endUpramp;
        const bool during_first_exp = ( TIME_1 < runTime ) &&
            ( runTime < TIME_2 );
        const bool after_peakpulse = startDownramp <= runTime;

        if ( before_preupramp )
            env = 0.;
        else if ( before_start )
        {
            env = AMP_1 * gauss( runTime - TIME_1 );
        }
        else if ( before_peakpulse )
        {
            const float_X ramp_when_peakpulse = extrapolate_expo(
                TIME_2,
                AMP_2,
                TIME_3,
                AMP_3,
                endUpramp
            ) / AMPLITUDE;

            if ( ramp_when_peakpulse > 0.5 )
            {
                log< picLog::PHYSICS >(
                    "Attention, the intensities of the laser upramp are very large! "
                    "The extrapolation of the last exponential to the time of "
                    "the peakpulse gives more than half of the amplitude of "
                    "the peak Gaussian. This is not a Gaussian at all anymore, "
                    "and physically very unplausible, check the params for misunderstandings!"
                );
            }

            env += AMPLITUDE * ( float_X( 1. ) - ramp_when_peakpulse ) *
                gauss( runTime - endUpramp );
            env += AMP_PREPULSE * gauss( runTime - TIME_PREPULSE );
            if ( during_first_exp )
                env += extrapolate_expo(
                    TIME_1,
                    AMP_1,
                    TIME_2,
                    AMP_2,
                    runTime
                );
            else
                env += extrapolate_expo(
                    TIME_2,
                    AMP_2,
                    TIME_3,
                    AMP_3,
                    runTime
                );
        }
        else if ( !after_peakpulse )
            env = AMPLITUDE;
    else // after startDownramp
            env = AMPLITUDE * gauss( runTime - startDownramp );
        return env;
    }

    HINLINE float3_X laserLongitudinal(uint32_t currentStep, float_X& phase)
    {
        float_X envelope;
        float3_X elong( float3_X::create( 0.0 ) );

        // a symmetric pulse will be initialized at position z=0 for
        // a time of RAMP_INIT * PULSE_LENGTH + LASER_NOFOCUS_CONSTANT = INIT_TIME.
        // we shift the complete pulse for the half of this time to start with
        // the front of the laser pulse.

        /* initialize the laser not in the first cell is equal to a negative shift
         * in time
         */
        const float_64 runTime = time_start_init - laserTimeShift +
            DELTA_T * currentStep;

        phase += float_X( w * runTime ) + LASER_PHASE;

        envelope = get_envelope( runTime );

        if( Polarisation == LINEAR_X )
        {
            elong.x() = float_X( envelope * ( math::sin( phase ) ) );
        }
        else if( Polarisation == LINEAR_Z )
        {
            elong.z() = float_X( envelope * ( math::sin( phase ) ) );
        }
        else if( Polarisation == CIRCULAR )
        {
            elong.x() = float_X( envelope / sqrt( 2.0 ) * ( math::sin( phase ) ) );
            elong.z() = float_X( envelope / sqrt( 2.0 ) * ( math::cos( phase ) ) );
        }

        return elong;
    }

    /**
     * takes the 3-component-vector elong of the E-field (computed for the
     * current timestep), and the x- and z-position and returns this elong
     * modulated by an isotropic Gaussian in transversal direction.
     * @param elong
     * @param phase
     * @param posX
     * @param posZ
     * @return
     */
    HDINLINE float3_X laserTransversal(float3_X elong, const float_X, const float_X posX, const float_X posZ)
    {

        const float_X exp_x = posX * posX / (W0_X * W0_X);
        const float_X exp_z = posZ * posZ / (W0_Z * W0_Z);

        return elong * math::exp( float_X( -1.0 ) * ( exp_x + exp_z ) );

    }

}
}

