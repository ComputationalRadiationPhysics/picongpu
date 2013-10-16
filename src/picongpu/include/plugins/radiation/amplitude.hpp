/**
 * Copyright 2013 Heiko Burau, René Widera, Richard Pausch
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

#include "complex.hpp"
#include "parameters.hpp"


class Amplitude
{
    /// class to store 3 complex numbers for the radiated amplitude
public:
    // constructor (real vector and complex phase)

  DINLINE Amplitude(vec2 vec, picongpu::float_X phase)
    {
        picongpu::float_X cosValue;
        picongpu::float_X sinValue;
	picongpu::math::sincos(phase, sinValue, cosValue);
        amp_x.euler(vec.x(), sinValue, cosValue);
        amp_y.euler(vec.y(), sinValue, cosValue);
        amp_z.euler(vec.z(), sinValue, cosValue);
    }

    // default constructor (initializes all values to zero)

    HDINLINE Amplitude(void)
    {

    }

    HDINLINE Amplitude(const numtype2 x_re, const numtype2 x_im, const numtype2 y_re, const numtype2 y_im, const numtype2 z_re, const numtype2 z_im)
      : amp_x(x_re, x_im), amp_y(y_re, y_im), amp_z(z_re, z_im)
    {
      
    }

    HDINLINE static Amplitude zero(void)
    {
        Amplitude result;
        result.amp_x = Complex::zero();
        result.amp_y = Complex::zero();
        result.amp_z = Complex::zero();
        return result;
    }

    // assign addition

    HDINLINE Amplitude& operator+=(const Amplitude& other)
    {
        amp_x += other.amp_x;
        amp_y += other.amp_y;
        amp_z += other.amp_z;
        return *this;
    }

    //  assign difference

    HDINLINE Amplitude& operator-=(const Amplitude& other)
    {
        amp_x -= other.amp_x;
        amp_y -= other.amp_y;
        amp_z -= other.amp_z;
        return *this;
    }

    // calc radiation from const*Amplitude^2

    HDINLINE numtype2 calc_radiation(void)
    {
        // returns \frac{d^2 I}{d \Omega d \omega}
        const numtype2 factor = 1.0 /
                (16. * util::cube(M_PI) * picongpu::EPS0 * picongpu::SPEED_OF_LIGHT); // SI factor radiation

        return factor * (amp_x.abs_square() + amp_y.abs_square() + amp_z.abs_square());
    }

    // debugging method: just returs real-x-value

    HDINLINE numtype2 debug(void)
    {
        return amp_x.get_real();
    }



private:
    Complex amp_x; // complex amplitude x-component
    Complex amp_y; // complex amplitude y-component
    Complex amp_z; // complex amplitude z-component


};

#include "mpi/GetMPI_StructAsArray.hpp"

namespace PMacc
{
namespace mpi
{

template<>
static MPI_StructAsArray getMPI_StructAsArray< ::Amplitude >()
{
    MPI_StructAsArray result = getMPI_StructAsArray< ::Complex::Type > ();
    result.sizeMultiplier *= 6;
    return result;
};

}//namespace mpi

}//namespace PMacc



