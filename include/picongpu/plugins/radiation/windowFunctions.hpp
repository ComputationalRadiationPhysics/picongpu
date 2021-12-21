/* Copyright 2014-2021 Richard Pausch
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

#include <pmacc/algorithms/math/defines/pi.hpp>

#include <cmath>

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            /* several window functions behind namespaces: */


            namespace radWindowFunctionTriangle
            {
                struct radWindowFunction
                {
                    /** 1D Window function according to the triangle window:
                     *
                     * x = position_x - L_x/2
                     * f(x) = {1+2x/L_x : (-L_x/2 <= x <= 0      )
                     *        {1-2x/L_x : (0      <= x <= +L_x/2 )
                     *        {0.0      : in any other case
                     *
                     * @param position_x = 1D position
                     * @param L_x        = length of the simulated area
                     *                     assuming that the simulation ranges
                     *                     from 0 to L_x in the chosen dimension
                     * @returns weighting factor to reduce ringing effects due to
                     *          sharp spacial boundaries
                     **/
                    HDINLINE float_X operator()(const float_X position_x, const float_X L_x) const
                    {
                        float_X x = position_x - float_X(0.5) * L_x;
                        return float_X(math::abs(x) <= float_X(0.5) * L_x)
                            * (float_X(1.0) - float_X(2.0) / L_x * math::abs(x));
                    }
                };
            } // namespace radWindowFunctionTriangle


            namespace radWindowFunctionHamming
            {
                struct radWindowFunction
                {
                    /** 1D Window function according to the Hamming window:
                     *
                     * x = position_x - L_x/2
                     * a = parameter of the Hamming window (ideal: 0.08)
                     * f(x) = {a+(1-a)*cos^2(pi*x/L_x)   : (-L_x/2 <= x <= +L_x/2 )
                     *        {0.0                       : in any other case
                     *
                     * @param position_x = 1D position
                     * @param L_x        = length of the simulated area
                     *                     assuming that the simulation ranges
                     *                     from 0 to L_x in the chosen dimension
                     * @returns weighting factor to reduce ringing effects due to
                     *          sharp spacial boundaries
                     **/
                    HDINLINE float_X operator()(const float_X position_x, const float_X L_x) const
                    {
                        const float_X x = position_x - L_x * float_X(0.5);
                        const float_X a = 0.08; /* ideal parameter: -43dB reduction */
                        const float_X cosinusValue = math::cos(pmacc::math::Pi<float_X>::value * x / L_x);
                        return float_X(math::abs(x) <= float_X(0.5) * L_x)
                            * (a + (float_X(1.0) - a) * cosinusValue * cosinusValue);
                    }
                };
            } // namespace radWindowFunctionHamming


            namespace radWindowFunctionTriplett
            {
                struct radWindowFunction
                {
                    /** 1D Window function according to the Triplett window:
                     *
                     * x      = position_x - L_x/2
                     * lambda = decay parameter of the Triplett window
                     * f(x) = {exp(-lambda*|x|)*cos^2(pi*x/L_x) : (-L_x/2 <= x <= +L_x/2 )
                     *        {0.0                              : in any other case
                     *
                     * @param position_x = 1D position
                     * @param L_x        = length of the simulated area
                     *                     assuming that the simulation ranges
                     *                     from 0 to L_x in the chosen dimension
                     * @returns weighting factor to reduce ringing effects due to
                     *          sharp spacial boundaries
                     **/
                    HDINLINE float_X operator()(const float_X position_x, const float_X L_x) const
                    {
                        const float_X x = position_x - L_x * float_X(0.5);
                        const float_X lambda = float_X(5.0) / L_x; /* larger is better, but too large means no data */
                        const float_X cosinusValue = math::cos(pmacc::math::Pi<float_X>::value * x / L_x);
                        return float_X(math::abs(x) <= float_X(0.5) * L_x)
                            * (math::exp(float_X(-1.0) * lambda * math::abs(x)) * cosinusValue * cosinusValue);
                    }
                };
            } // namespace radWindowFunctionTriplett


            namespace radWindowFunctionGauss
            {
                struct radWindowFunction
                {
                    /** 1D Window function according to the Gauss window:
                     *
                     * x     = position_x - L_x/2
                     * sigma = standard deviation of the Gauss window
                     * f(x) = {exp(-0.5*x^2/sigma^2)   : (-L_x/2 <= x <= +L_x/2 )
                     *        {0.0                     : in any other case
                     *
                     * @param position_x = 1D position
                     * @param L_x        = length of the simulated area
                     *                     assuming that the simulation ranges
                     *                     from 0 to L_x in the chosen dimension
                     * @returns weighting factor to reduce ringing effects due to
                     *          sharp spacial boundaries
                     **/
                    HDINLINE float_X operator()(const float_X position_x, const float_X L_x) const
                    {
                        const float_X x = position_x - L_x * float_X(0.5);
                        const float_X sigma = float_X(0.4) * L_x; /* smaller is better, but too small means no data */
                        const float_X relativePosition = x / sigma; /* optimization */
                        return float_X(math::abs(x) <= float_X(0.5) * L_x)
                            * (math::exp(float_X(-0.5) * relativePosition * relativePosition));
                    }
                };
            } // namespace radWindowFunctionGauss


            namespace radWindowFunctionNone
            {
                struct radWindowFunction
                {
                    /** 1D Window function according to the no window:
                     *
                     * f(position_x) = always 1.0
                     *
                     * @param position_x = 1D position
                     * @param L_x        = length of the simulated area
                     *                     assuming that the simulation ranges
                     *                     from 0 to L_x in the chosen dimension
                     * @returns 1.0
                     **/
                    HDINLINE float_X operator()(const float_X position_x, const float_X L_x) const
                    {
                        return float_X(1.0);
                    }
                };
            } // namespace radWindowFunctionNone


        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
