/**
 * Copyright 2013 Heiko Burau, Rene Widera, Richard Pausch
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

#include<cmath>
#include <iostream>

#pragma once

#include "utilities.hpp"

template<typename T>
class Complex_T
{
    /// a complex number class
public:

    typedef T Type;
    // constructor (real, imaginary)

    HDINLINE Complex_T(T real, T imaginary = 0.0)
    : real(real), imaginary(imaginary)
    {
    };

    // default constructor ( ! no initialization of data ! )

    HDINLINE Complex_T(void)
    {
    };

    HDINLINE static Complex_T<T> zero(void)
    {
        return Complex_T<T > ().euler(0, 0, 1);
    };

    // set complex number by using Euler's formula

    HDINLINE Complex_T euler(T magnitude, const T& phase)
    {
        real = magnitude * picongpu::math::cos(picongpu::precisionCast<picongpu::float_X>(phase));
        imaginary = magnitude * picongpu::math::sin(picongpu::precisionCast<picongpu::float_X>(phase));
        return *this;
    }

    HDINLINE Complex_T euler(T magnitude, const T& sinValue, const T& cosValue)
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

    HDINLINE Complex_T operator+(const Complex_T& other)
    {
        return Complex_T(real + other.real, imaginary + other.imaginary);
    }
	
	HDINLINE Complex_T operator+(const Complex_T& lhs, const Complex_T& rhs)
    {
        return Complex_T(lhs.real + rhs.real, lhs.imaginary + rhs.imaginary);
    }

	HDINLINE Complex_T operator+(const Complex_T& lhs, const T& rhs)
    {
        return Complex_T(lhs.real + rhs, lhs.imaginary);
    }
	
	HDINLINE Complex_T operator+(const T& lhs, const Complex_T& rhs)
    {
        return Complex_T(lhs + rhs.real, rhs.imaginary);
    }
	
    // difference

    HDINLINE Complex_T operator-(const Complex_T& other)
    {
        return Complex_T(real - other.real, imaginary - other.imaginary);
    }
	
	HDINLINE Complex_T operator-(const Complex_T& lhs, const Complex_T& rhs)
    {
        return Complex_T(lhs.real - rhs.real, lhs.imaginary - rhs.imaginary);
    }

	HDINLINE Complex_T operator-(const Complex_T& lhs, const T& rhs)
    {
        return Complex_T(lhs.real - rhs, lhs.imaginary);
    }
	
	HDINLINE Complex_T operator-(const T& lhs, const Complex_T& rhs)
    {
        return Complex_T(lhs - rhs.real, -rhs.imaginary);
    }
	
    // multiplication

    HDINLINE Complex_T operator*(const Complex_T& other)
    {
        return Complex_T(real * other.real - imaginary * other.imaginary,
                         imaginary * other.real + real * other.imaginary);
    }

	HDINLINE Complex_T operator*(const Complex_T& lhs, const Complex_T& rhs)
    {
        return Complex_T(lhs.real * rhs.real - lhs.imaginary * rhs.imaginary,
                         lhs.imaginary * rhs.real + lhs.real * rhs.imaginary);
    }
	
	HDINLINE Complex_T operator*(const Complex_T& lhs, const T& rhs)
    {
        return Complex_T(lhs.real * rhs, lhs.imaginary * rhs);
    }
	
	HDINLINE Complex_T operator*(const T& lhs, const Complex_T& rhs)
    {
        return Complex_T(lhs * rhs.real, lhs * rhs.imaginary);
    }
	
	// Division
	
	HDINLINE Complex_T operator/(const Complex_T& lhs, const T& rhs)
    {
        return Complex_T(lhs.real / rhs, lhs.imaginary / rhs);
    }
	
	HDINLINE Complex_T operator/(const T& lhs, const Complex_T& rhs)
    {
        return Complex_T(lhs * rhs.real/(rhs.real*rhs.real+rhs.imaginary*rhs.imaginary),
                         -lhs * rhs.imaginary/( util::square(rhs.real)+util::square(rhs.imaginary) ));
    }

	HDINLINE Complex_T operator/(const Complex_T& lhs, const Complex_T& rhs)
    {
        return lhs*Complex_T(rhs.real/(rhs.real*rhs.real+rhs.imaginary*rhs.imaginary),
                            -rhs.imaginary/( util::square(rhs.real)+util::square(rhs.imaginary) ));
    }
	
	// Conversion from scalar (assignment)
	
	HDINLINE Complex_T& operator=(const T& other)
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

    // absolute value

    HDINLINE T abs(void)
    {
        return sqrtf(util::square(real) + util::square(imaginary));
    }

    // square of absolute value

    HDINLINE T abs_square(void)
    {
        return util::square(real) + util::square(imaginary);
    }

    // real part

    HDINLINE T get_real(void)
    {
        return real;
    }

    // imaginary part

    HDINLINE T get_imag(void)
    {
        return imaginary;
    }


private:
    __align__(sizeof (T)) T real; // real part
    __align__(sizeof (T)) T imaginary; // imaginary part


};


typedef Complex_T<numtype2> Complex; // a complex number class using numtype as data-type


