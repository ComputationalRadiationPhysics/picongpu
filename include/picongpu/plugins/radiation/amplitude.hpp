/* Copyright 2013-2019 Heiko Burau, Rene Widera, Richard Pausch, Alexander Debus
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
#include <pmacc/math/Complex.hpp>
#include "parameters.hpp"
#include <pmacc/mpi/GetMPI_StructAsArray.hpp>


namespace picongpu
{
/** class to store 3 complex numbers for the radiated amplitude
 */
class Amplitude
{
public:
  using complex_64 = pmacc::math::Complex< picongpu::float_64 >;

  /* number of scalar components in Amplitude = 3 (3D) * 2 (complex) = 6 */
  static constexpr uint32_t numComponents = uint32_t(3) * uint32_t(sizeof(complex_64) / sizeof(typename complex_64::type));

  /** constructor
   *
   * Arguments:
   * - vector_64: real 3D vector
   * - float: complex phase */
  DINLINE Amplitude(vector_64 vec, picongpu::float_X phase)
  {
      picongpu::float_X cosValue;
      picongpu::float_X sinValue;
      picongpu::math::sincos(phase, sinValue, cosValue);
      amp_x=picongpu::math::euler(vec.x(), picongpu::precisionCast<picongpu::float_64>(sinValue), picongpu::precisionCast<picongpu::float_64>(cosValue) );
      amp_y=picongpu::math::euler(vec.y(), picongpu::precisionCast<picongpu::float_64>(sinValue), picongpu::precisionCast<picongpu::float_64>(cosValue) );
      amp_z=picongpu::math::euler(vec.z(), picongpu::precisionCast<picongpu::float_64>(sinValue), picongpu::precisionCast<picongpu::float_64>(cosValue) );
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
  HDINLINE Amplitude(const picongpu::float_64 x_re, const picongpu::float_64 x_im,
                     const picongpu::float_64 y_re, const picongpu::float_64 y_im,
                     const picongpu::float_64 z_re, const picongpu::float_64 z_im)
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
   * Returns: \f$\frac{d^2 I}{d \Omega d \omega} = const*Amplitude^2\f$ */
  HDINLINE picongpu::float_64 calc_radiation(void)
  {
      // const SI factor radiation
      const picongpu::float_64 factor = 1.0 /
        (16. * util::cube(pmacc::algorithms::math::Pi< picongpu::float_64 >::value) * picongpu::EPS0 * picongpu::SPEED_OF_LIGHT);

      return factor * (picongpu::math::abs2(amp_x) + picongpu::math::abs2(amp_y) + picongpu::math::abs2(amp_z));
  }


  /** debugging method
   *
   * Returns: real-x-value */
  HDINLINE picongpu::float_64 debug(void)
  {
      return amp_x.get_real();
  }


private:
  complex_64 amp_x; // complex amplitude x-component
  complex_64 amp_y; // complex amplitude y-component
  complex_64 amp_z; // complex amplitude z-component

};
} // namespace picongpu

namespace pmacc
{
namespace mpi
{

  /** implementation of MPI transaction on Amplitude class */
  template<>
  MPI_StructAsArray getMPI_StructAsArray< picongpu::Amplitude >()
  {
      MPI_StructAsArray result = getMPI_StructAsArray< picongpu::Amplitude::complex_64::type > ();
      result.sizeMultiplier *= picongpu::Amplitude::numComponents;
      return result;
  };

} // namespace mpi
} // namespace pmacc
