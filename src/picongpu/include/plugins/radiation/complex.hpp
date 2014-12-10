/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera, Richard Pausch, Alexander Debus
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

#include "utilities.hpp"

#include <cmath>
#include <iostream>

template<typename T_Type>
class Complex_T
{
    /// a complex number class
public:

    typedef T_Type Type;
    // constructor (real, imaginary)
    HDINLINE Complex_T(Type real, Type imaginary = 0.0)
    : real(real), imaginary(imaginary)
    {
    };

    // default constructor ( ! no initialization of data ! )
    HDINLINE Complex_T(void)
    {
    };

    HDINLINE static Complex_T<Type> zero(void)
    {
        return Complex_T<Type > ().euler(0, 0, 1);
    };

    // set complex number by using Euler's formula
    HDINLINE Complex_T euler(Type magnitude, const Type& phase)
    {
        real = magnitude * picongpu::math::cos(picongpu::precisionCast<picongpu::float_X>(phase));
        imaginary = magnitude * picongpu::math::sin(picongpu::precisionCast<picongpu::float_X>(phase));
        return *this;
    }

    HDINLINE Complex_T euler(Type magnitude, const Type& sinValue, const Type& cosValue)
    {
        real = magnitude * cosValue;
        imaginary = magnitude * sinValue;
        return *this;
    }

    // print complex number to screen (host only)
    HINLINE void print(void)
    {
        std::cout << " ( " << real << " +i " << imaginary << " ) " << std::endl;
    }

    // addition
    HDINLINE friend Complex_T operator+(const Complex_T& lhs, const Complex_T& rhs)
    {
        return Complex_T(lhs.real + rhs.real, lhs.imaginary + rhs.imaginary);
    }

    HDINLINE friend Complex_T operator+(const Complex_T& lhs, const Type& rhs_scalar)
    {
        return Complex_T(lhs.real + rhs_scalar, lhs.imaginary);
    }

    HDINLINE friend Complex_T operator+(const Type& lhs_scalar, const Complex_T& rhs)
    {
        return Complex_T(lhs_scalar + rhs.real, rhs.imaginary);
    }

    // substraction
    HDINLINE friend Complex_T operator-(const Complex_T& lhs, const Complex_T& rhs)
    {
        return Complex_T(lhs.real - rhs.real, lhs.imaginary - rhs.imaginary);
    }

    HDINLINE friend Complex_T operator-(const Complex_T& lhs, const Type& rhs)
    {
        return Complex_T(lhs.real - rhs, lhs.imaginary);
    }

    HDINLINE friend Complex_T operator-(const Type& lhs_scalar, const Complex_T& rhs)
    {
        return Complex_T(lhs_scalar - rhs.real, -rhs.imaginary);
    }

    // multiplication
    HDINLINE friend Complex_T operator*(const Complex_T& lhs, const Complex_T& rhs)
    {
        return Complex_T(lhs.real * rhs.real - lhs.imaginary * rhs.imaginary,
                         lhs.imaginary * rhs.real + lhs.real * rhs.imaginary);
    }

    HDINLINE friend Complex_T operator*(const Complex_T& lhs, const Type& rhs_scalar)
    {
        return Complex_T(lhs.real * rhs_scalar, lhs.imaginary * rhs_scalar);
    }

    HDINLINE friend Complex_T operator*(const Type& lhs_scalar, const Complex_T& rhs)
    {
        return Complex_T(lhs_scalar * rhs.real, lhs_scalar * rhs.imaginary);
    }

    // Division
    HDINLINE friend Complex_T operator/(const Complex_T& lhs, const Type& rhs_scalar)
    {
        return Complex_T(lhs.real / rhs_scalar, lhs.imaginary / rhs_scalar);
    }

    HDINLINE friend Complex_T operator/(const Type& lhs_scalar, const Complex_T& rhs)
    {
        return Complex_T( lhs_scalar * rhs.real / ( util::square(rhs.real) + util::square(rhs.imaginary) ),
                         -lhs_scalar * rhs.imaginary / ( util::square(rhs.real) + util::square(rhs.imaginary) ) );
    }

    HDINLINE friend Complex_T operator/(const Complex_T& lhs, const Complex_T& rhs)
    {
        return lhs*Complex_T( rhs.real / ( util::square(rhs.real) + util::square(rhs.imaginary) ),
                             -rhs.imaginary / ( util::square(rhs.real) + util::square(rhs.imaginary) ) );
    }

    HDINLINE static Complex_T<Type > csqrt(const Complex_T<Type >& other)
    {
        if (other.real<=0.0 && other.imaginary==0.0) {
            return Complex_T<Type > (0.0,sqrt(-other.real));
        }
        else {
            return sqrt(other.abs_64())*(other+other.abs_64())/(other+other.abs_64()).abs_64();
        }
    }

    HDINLINE static Complex_T<Type > cpow(const Complex_T<Type >& other, const Type& exponent)
    {
        return pow( other.abs_64(), exponent) *
               cexp( Complex_T<Type >( 0.,1. ) * other.arg() * exponent );
    }

    // Complex exponential function
    HDINLINE static Complex_T<Type > cexp(const Complex_T<Type >& other)
    {
        return Complex_T<Type > ().euler(1.0,other.imaginary)*expf(other.real);
    }

    // Conversion from scalar (assignment)
    HDINLINE Complex_T& operator=(const Type& other)
    {
        real = other;
        return *this;
    }

    // Assignment operator
    HDINLINE Complex_T& operator=(const Complex_T& other)
    {
        real = other.real;
        imaginary = other.imaginary;
        return *this;
    }

    // assign addition
    HDINLINE Complex_T& operator+=(const Complex_T& other)
    {
        real += other.real;
        imaginary += other.imaginary;
        return *this;
    }

    // assign difference
    HDINLINE Complex_T& operator-=(const Complex_T& other)
    {
        real -= other.real;
        imaginary -= other.imaginary;
        return *this;
    }

    // assign multiplication
    HDINLINE Complex_T& operator *=(const Complex_T& other)
    {
        *this = *this * other;
        return *this;
    }

    // Absolute value
    HDINLINE Type abs_64(void) const
    {
        // For the complex square root cexp(), the slower sqrt() is required to avoid infinities.
        return sqrt(util::square(real) + util::square(imaginary));
    }

    // Absolute value: Faster float version for Richards radiation plugin.
    HDINLINE Type abs(void) const
    {
        // For the complex square root cexp(), the slower sqrt() is required to avoid infinities.
        return sqrtf(util::square(real) + util::square(imaginary));
    }

    // Phase of complex number (Note: Branchcut running from -infinity to 0)
    HDINLINE Type arg(void) const
    {
        if (real==0.0 && imaginary==0.0) return 0.0;
        else if (real==0.0 && imaginary>0.0) return picongpu::PI/2;
        else if (real==0.0 && imaginary<0.0) return -picongpu::PI/2;
        else if (real<0.0 && imaginary==0.0) return picongpu::PI;
        else return atan2(imaginary,real);
    }

    // square of absolute value
    HDINLINE Type abs_square(void) const
    {
        return util::square(real) + util::square(imaginary);
    }

    // real part
    HDINLINE Type get_real(void) const
    {
        return real;
    }

    // imaginary part
    HDINLINE Type get_imag(void) const
    {
        return imaginary;
    }


  private:
    __align__(sizeof (Type)) Type real; // real part
    __align__(sizeof (Type)) Type imaginary; // imaginary part

};


// a complex number class using numtype as data-type
typedef Complex_T<numtype2> Complex;

namespace picongpu
{
    // a complex number class using float_64 as data-type
    typedef Complex_T<float_64> Complex_64;
}
