/**
 * Copyright 2013, 2015 Heiko Burau, Rene Widera, Richard Pausch
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

namespace picongpu
{

  namespace radFormFactor_CIC_3D
  {
    class radFormFactor
    {
    public:
      HDINLINE radFormFactor(void)
      { }

      HDINLINE float_X operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
      {

    /* Form Factor for CIC charge distribution of N discrete electrons:
     * | \mathcal{F} |^2 = N + (N*N - N) * sinc^2(n_x * L_x * \omega) * sinc^2(n_y * L_y * \omega) * sinc^2(n_z * L_z * \omega)
     *
     * with observation direction (unit vector) \vec{n} = (n_x, n_y, n_z)
     * and with: N     = weighting
     *           omega = frequency
     *           L_d   = the size of the CIC-particle / cell in dimension d
     *
     * the Form Factor: sqrt( | \mathcal{F} |^2 ) will be returned
     */

    return sqrt(N + (N*N - N) * util::square(
                         math::sinc( observer_unit_vec.x() * CELL_WIDTH/(SPEED_OF_LIGHT*2)  * omega) *
                         math::sinc( observer_unit_vec.y() * CELL_HEIGHT/(SPEED_OF_LIGHT*2) * omega) *
                         math::sinc( observer_unit_vec.z() * CELL_DEPTH/(SPEED_OF_LIGHT*2)  * omega)
                          )
            );

      }
    private:

    };
  } // end namespace: radFormFactor_CIC_3D

  namespace radFormFactor_CIC_1Dy
  {
    class radFormFactor
    {
    public:
      HDINLINE radFormFactor(void)
      { }

      HDINLINE float_X operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
      {

    /* Form Factor for 1D CIC charge distribution of N discrete electrons:
     */

    return sqrt(N + (N*N - N) * util::square(math::sinc( CELL_HEIGHT/(SPEED_OF_LIGHT*2) * omega) ) );

      }
    private:

    };
  } // end namespace: radFormFactor_CIC_1Dy



  namespace radFormFactor_incoherent
  {
    class radFormFactor
    {
    public:
      HDINLINE radFormFactor(void)
      { }

      HDINLINE float_X operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
      {

    /* Form Factor for 1D CIC charge distribution of N discrete electrons:
     */

    return sqrt(N);

      }
    private:

    };
  } // end namespace: radFormFactor_incoherent




}
