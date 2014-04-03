/**
 * Copyright 2014 Richard Pausch
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

#include<cmath>

namespace picongpu
{

  /* several window functions behind namespaces: */


  namespace radWindowFunctionRectangle
  {
    struct radWindowFunction
    {
      /** 1D Window function according to the rectangle window:
       *
       * f(position_x) = {1.0     : (0 <= position_x <= L_x )
       *                 {0.0     : in any other case
       *
       * @param position_x = 1D position
       * @param L_x        = length of the simulated area
       *                     assuming that the simulation ranges
       *                     from 0 to L_x in the choosen dimension
       * @returns weighting factor to reduce ringing effects due to
       *          sharp spacial boundaries
       **/
      HDINLINE float_X operator()(const float_X position_x, const float_X L_x) const
      {
	/* an optimized formula is implemented 
	 * 
	 * transform position to make box symetric:
	 * x_prime = position_x - 1/2 * L_x
	 * 
	 * then: f(x_position) = f(x_prime)
	 * f(x_prime) = { 1.0     : -L_x/2 <= x_prime <= +L_x/2
	 *              { 0.0     : in any other case
	 */
	const float_X x_prime = position_x - L_x*float_X(0.5);
	return float_X(math::abs(x_prime) <= float_X(0.5) * L_x);
      }
    };
  } /* namespace radWindowFunctionRectangle */



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
       *                     from 0 to L_x in the choosen dimension
       * @returns weighting factor to reduce ringing effects due to
       *          sharp spacial boundaries
       **/
      HDINLINE float_X operator()(const float_X position_x, const float_X L_x) const
      {
	float_X x = position_x - float_X(0.5)*L_x;
	return float_X(math::abs(x) <= float_X(0.5)*L_x * (float_X(1.0)-
				     float_X(2.0)/L_x * math::abs(x) ));
      }
    };
  } /* namespace radWindowFunctionTriangle */


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
       *                     from 0 to L_x in the choosen dimension
       * @returns weighting factor to reduce ringing effects due to
       *          sharp spacial boundaries
       **/
      HDINLINE float_X operator()(const float_X position_x, const float_X L_x) const
      {
	const float_X x = position_x - L_x*float_X(0.5);
	const float_X a = 0.08; /* ideal parameter: -43dB reduction */
	const float_X cosinusValue = math::cos(M_PI*x/L_x);
	return a + (float_X(1.0)-a)*cosinusValue*cosinusValue;
      }
    };
  } /* namespace radWindowFunctionHamming */








}  /* namespace picongpu */

