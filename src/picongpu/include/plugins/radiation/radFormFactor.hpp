/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera, Richard Pausch
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
      /* Form Factor for CIC charge distribution of N discrete electrons:
       * | \mathcal{F} |^2 = N + (N*N - N) * sinc^2(n_x * L_x * \omega) * sinc^2(n_y * L_y * \omega) * sinc^2(n_z * L_z * \omega)
       *
       * with observation direction (unit vector) \vec{n} = (n_x, n_y, n_z)
       * and with: N     = weighting
       *           omega = frequency
       *           L_d   = the size of the CIC-particle / cell in dimension d
       *
       * @param N = macro particle weighting
       * @param omega = frequency at which to calculate the  form factor
       * @param observer_unit_vec = observation direction
       * @return the Form Factor: sqrt( | \mathcal{F} |^2 )
       */
      HDINLINE float_X operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
      {
        //util::Pow<float_X, 1> pow;
        const float_X sincX = util::pow( math::sinc( observer_unit_vec.x() * CELL_WIDTH / ( SPEED_OF_LIGHT * float_X( 2.0 ) ) * omega ), 1 );
        const float_X sincY = util::pow( math::sinc( observer_unit_vec.y() * CELL_HEIGHT / ( SPEED_OF_LIGHT * float_X( 2.0 ) ) * omega ), 1 );
        const float_X sincZ = util::pow( math::sinc( observer_unit_vec.z() * CELL_DEPTH / ( SPEED_OF_LIGHT * float_X( 2.0 ) ) * omega ), 1 );
          return math::sqrt(
              N + ( N * N - N ) * util::square( sincX * sincY * sincZ )
          ); // math::sqrt
      }
    };
  } // radFormFactor_CIC_3D


  namespace radFormFactor_CIC_1Dy
  {
    class radFormFactor
    {
      /* Form Factor for 1-d CIC charge distribution iy y of N discrete electrons:
       * | \mathcal{F} |^2 = N + (N*N - N) * sinc^2(n_y * L_y * \omega)
       *
       * with observation direction (unit vector) \vec{n} = (n_x, n_y, n_z)
       * and with: N     = weighting
       *           omega = frequency
       *           L_d   = the size of the CIC-particle / cell in dimension d
       *
       * @param N = macro particle weighting
       * @param omega = frequency at which to calculate the  form factor
       * @param observer_unit_vec = observation direction
       * @return the Form Factor: sqrt( | \mathcal{F} |^2 )
       */
      HDINLINE float_X operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
      {
          return math::sqrt(
              N + ( N * N - N ) * util::square(
                  math::sinc( CELL_HEIGHT / ( SPEED_OF_LIGHT * float_X( 2.0 ) ) * omega )
              )
          );
      }
    };
  } // radFormFactor_CIC_1Dy


  namespace radFormFactor_TSC_3D
  {
    class radFormFactor
    {
    public:
      /* Form Factor for TSC charge distribution of N discrete electrons:
       *
       * with observation direction (unit vector) \vec{n} = (n_x, n_y, n_z)
       * and with: N     = weighting
       *           omega = frequency
       *           L_d   = the size of the cell in dimension d
       *
       * @param N = macro particle weighting
       * @param omega = frequency at which to calculate the  form factor
       * @param observer_unit_vec = observation direction
       * @return the Form Factor: sqrt( | \mathcal{F} |^2 )
       */
      HDINLINE float_X operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
      {
        //util::Pow<float_X, 2> pow;
        const float_X sincX = util::pow( math::sinc( observer_unit_vec.x() * CELL_WIDTH / ( SPEED_OF_LIGHT * float_X( 2.0 ) ) * omega ), 2 );
        const float_X sincY = util::pow( math::sinc( observer_unit_vec.y() * CELL_HEIGHT / ( SPEED_OF_LIGHT * float_X( 2.0 ) ) * omega ), 2 );
        const float_X sincZ = util::pow( math::sinc( observer_unit_vec.z() * CELL_DEPTH / ( SPEED_OF_LIGHT * float_X( 2.0 ) ) * omega ), 2 );
          return math::sqrt(
              N + ( N * N - N ) * util::square( sincX * sincY * sincZ )
          ); // math::sqrt
      }
    };
  } // radFormFactor_TSC_3D


  namespace radFormFactor_PCS_3D
  {
    class radFormFactor
    {
    public:
      /* Form Factor for PCS charge distribution of N discrete electrons:
       *
       * with observation direction (unit vector) \vec{n} = (n_x, n_y, n_z)
       * and with: N     = weighting
       *           omega = frequency
       *           L_d   = the size of the cell in dimension d
       *
       * @param N = macro particle weighting
       * @param omega = frequency at which to calculate the  form factor
       * @param observer_unit_vec = observation direction
       * @return the Form Factor: sqrt( | \mathcal{F} |^2 )
       */
      HDINLINE float_X operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
      {
        //util::Pow<float_X, 3> pow;
        const float_X sincX = util::pow( math::sinc( observer_unit_vec.x() * CELL_WIDTH / ( SPEED_OF_LIGHT * float_X( 2.0 ) ) * omega ), 3 );
        const float_X sincY = util::pow( math::sinc( observer_unit_vec.y() * CELL_HEIGHT / ( SPEED_OF_LIGHT * float_X( 2.0 ) ) * omega ), 3 );
        const float_X sincZ = util::pow( math::sinc( observer_unit_vec.z() * CELL_DEPTH / ( SPEED_OF_LIGHT * float_X( 2.0 ) ) * omega ), 3 );
          return math::sqrt(
              N + ( N * N - N ) * util::square( sincX * sincY * sincZ )
          ); // math::sqrt
      }
    };
  } // radFormFactor_PCS_3D



  namespace radFormFactor_Gauss_spherical
  {
    class radFormFactor
    {
    public:
      /** Form Factor for point-symmetric Gauss-shaped charge distribution of N discrete electrons:
        * <rho(r)> = N*q_e* 1/sqrt(2*pi*sigma^2) * exp(-0.5 * r^2/sigma^2)
        * with sigma = 0.5*c/delta_t (0.5 because sigma is defined around center)
        *
        * @param N = macro particle weighting
        * @param omega = frequency at which to calculate the  form factor
        * @param observer_unit_vec = observation direction
        * @return the Form Factor: sqrt( | \mathcal{F} |^2 )
        */
      HDINLINE float_X operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
      {
          /* currently a fixed sigma of DELTA_T * c is used to describe the distribution - might become a parameter */
          return math::sqrt(
              N + ( N * N - N ) * util::square(
                  math::exp( float_X( -0.5 ) * util::square( omega * float_X( 0.5 ) * DELTA_T ) )
              )
          );
      }
    };
  } // radFormFactor_Gauss_spherical


  namespace radFormFactor_Gauss_cell
  {
    class radFormFactor
    {
    public:
      /** Form Factor for per-dimension Gauss-shaped charge distribution of N discrete electrons:
        * <rho(r)> = N*q_e* product[d={x,y,z}](1/sqrt(2*pi*sigma_d^2) * exp(-0.5 * d^2/sigma_d^2))
        * with sigma_d = 0.5*cell_width_d*n_d
        *
        * @param N = macro particle weighting
        * @param omega = frequency at which to calculate the  form factor
        * @param observer_unit_vec = observation direction
        * @return the Form Factor: sqrt( | \mathcal{F} |^2 )
        */
      HDINLINE float_X operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
      {
        return math::sqrt(
             N + ( N * N - N ) * util::square(
                 math::exp(
                     float_X( -0.5 ) * (
                         util::square( observer_unit_vec.x() * CELL_WIDTH / ( SPEED_OF_LIGHT * float_X(2.0) ) * omega ) +
                         util::square( observer_unit_vec.y() * CELL_HEIGHT / ( SPEED_OF_LIGHT * float_X(2.0) ) * omega ) +
                         util::square( observer_unit_vec.z() * CELL_DEPTH / ( SPEED_OF_LIGHT * float_X(2.0) ) * omega )
                     )
                 )
             )
        );
      }
    };
  } // radFormFactor_Gauss_cell



  namespace radFormFactor_incoherent
  {
    class radFormFactor
    {
    public:
      /** Form Factor for an incoherent charge distribution:
        *
        * @param N = macro particle weighting
        * @param omega = frequency at which to calculate the  form factor
        * @param observer_unit_vec = observation direction
        * @return the Form Factor: sqrt( | \mathcal{F} |^2 == sqrt(weighting)
        */
      HDINLINE float_X operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
      {
        return math::sqrt( N );

      }
    };
  } // radFormFactor_incoherent

}
