/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Richard Pausch, Alexander Debus
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

#include "math/Complex.hpp"
#include "parameters.hpp"
#include "mpi/GetMPI_StructAsArray.hpp"

typedef PMacc::math::Complex<numtype2> complex_64;

/** class to store 3 complex numbers for the radiated amplitude
 */
class Amplitude
{
public:
  // number of scalars of type numtype2 in Amplitude = 3 (3D) * 2 (complex) = 6
  static const uint numComponents = 3 * sizeof(complex_64) / sizeof(numtype2);

  /** constructor 
   * 
   * Arguments:
   * - vec2: real 3D vector 
   * - float: complex phase */
  DINLINE Amplitude(vec2 vec, picongpu::float_X phase)
  {
      picongpu::float_X cosValue;
      picongpu::float_X sinValue;
      picongpu::math::sincos(phase, sinValue, cosValue);
      amp_x=PMacc::algorithms::math::euler(vec.x(), picongpu::precisionCast<numtype2>(sinValue), picongpu::precisionCast<numtype2>(cosValue) );
      amp_y=PMacc::algorithms::math::euler(vec.y(), picongpu::precisionCast<numtype2>(sinValue), picongpu::precisionCast<numtype2>(cosValue) );
      amp_z=PMacc::algorithms::math::euler(vec.z(), picongpu::precisionCast<numtype2>(sinValue), picongpu::precisionCast<numtype2>(cosValue) );
  }


  /** default constructor 
   * 
   * \warning does not initialize values! */
  HDINLINE Amplitude(void)
  {

  }


  /** constructor 
   * 
   * Arguments:
   * - 6x float: Re(x), Im(x), Re(y), Im(y), Re(z), Im(z) */
  HDINLINE Amplitude(const numtype2 x_re, const numtype2 x_im, 
                     const numtype2 y_re, const numtype2 y_im, 
                     const numtype2 z_re, const numtype2 z_im)
      : amp_x(x_re, x_im), amp_y(y_re, y_im), amp_z(z_re, z_im)
  {

  }


  /** returns a zero amplitude vector
   *
   * used to initialize amplitudes to zero */
  HDINLINE static Amplitude zero(void)
  {
      Amplitude result;
      result.amp_x = complex_64::zero();
      result.amp_y = complex_64::zero();
      result.amp_z = complex_64::zero();
      return result;
  }

  /** assign addition */
  HDINLINE Amplitude& operator+=(const Amplitude& other)
  {
      amp_x += other.amp_x;
      amp_y += other.amp_y;
      amp_z += other.amp_z;
      return *this;
  }


  /** assign difference */
  HDINLINE Amplitude& operator-=(const Amplitude& other)
  {
      amp_x -= other.amp_x;
      amp_y -= other.amp_y;
      amp_z -= other.amp_z;
      return *this;
  }


  /** calculate radiation from *this amplitude
   *
   * Returns: \frac{d^2 I}{d \Omega d \omega} = const*Amplitude^2 */
  HDINLINE numtype2 calc_radiation(void)
  {
      // const SI factor radiation
      const numtype2 factor = 1.0 /
        (16. * util::cube(M_PI) * picongpu::EPS0 * picongpu::SPEED_OF_LIGHT); 

      return factor * (PMacc::algorithms::math::abs2(amp_x) + PMacc::algorithms::math::abs2(amp_y) + PMacc::algorithms::math::abs2(amp_z));
  }


  /** debugging method
   * 
   * Returns: real-x-value */
  HDINLINE numtype2 debug(void)
  {
      return amp_x.get_real();
  }


private:
  complex_64 amp_x; // complex amplitude x-component
  complex_64 amp_y; // complex amplitude y-component
  complex_64 amp_z; // complex amplitude z-component

};


namespace PMacc
{
namespace mpi
{

  /** implementation of MPI transaction on Amplitude class */
  template<>
  MPI_StructAsArray getMPI_StructAsArray< ::Amplitude >()
  {
      MPI_StructAsArray result = getMPI_StructAsArray< complex_64::type > ();
      result.sizeMultiplier *= Amplitude::numComponents;
      return result;
  };

}//namespace mpi
}//namespace PMacc



