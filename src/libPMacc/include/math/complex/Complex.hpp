/**
 * Copyright 2015 Alexander Debus
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
 /*
#include <builtin_types.h>
#include <cuda_runtime.h>
#include <boost/static_assert.hpp>
#include <boost/mpl/size.hpp>
#include <types.h>
*/

namespace PMacc
{
namespace math
{

/** A complex number class */
template<typename Type>
struct Complex : private __align__(sizeof(Type)) Type real, __align__(sizeof(Type)) Type imaginary
{
    typedef Type T_Type;

    // constructor (real, imaginary)
    HDINLINE Complex(Type real, Type imaginary = 0.0) : real(real), imaginary(imaginary);
    
    // constructor (Complex<T_OtherType>)
    template<typename OtherType>
    HDINLINE explicit Complex(const Complex<OtherType >& other) : real( static_cast<Type> (other.get_real()) ), imaginary( static_cast<Type> (other.get_imag()) );

    // default constructor ( ! no initialization of data ! )
    HDINLINE Complex(void) { };

    // Conversion from scalar (assignment)
    HDINLINE Complex& operator=(const Type& other)
    {
        real = other;
        return *this;
    }

    // Assignment operator
    HDINLINE Complex& operator=(const Complex& other)
    {
        real = other.real;
        imaginary = other.imaginary;
        return *this;
    }

    // assign addition
    HDINLINE Complex& operator+=(const Complex& other)
    {
        real += other.real;
        imaginary += other.imaginary;
        return *this;
    }

    // assign difference
    HDINLINE Complex& operator-=(const Complex& other)
    {
        real -= other.real;
        imaginary -= other.imaginary;
        return *this;
    }

    // assign multiplication
    HDINLINE Complex& operator *=(const Complex& other)
    {
        *this = *this * other;
        return *this;
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

};

/** Addition operators */

template<typename Type>
HDINLINE Complex<Type>
operator+(const Complex<Type>& lhs, const Complex<Type>& rhs)
{
    return Complex<Type>(lhs.real + rhs.real, lhs.imaginary + rhs.imaginary);
}

template<typename Type>
HDINLINE Complex<Type>
operator+(const Complex<Type>& lhs, const Type& rhs)
{
    return Complex<Type>(lhs.real + rhs, lhs.imaginary);
}

template<typename Type>
HDINLINE Complex<Type>
operator+(const Type& lhs, const Complex<Type>& rhs)
{
    return Complex<Type>(lhs + rhs.real, rhs.imaginary);
}

/** Substraction operators */

template<typename Type>
HDINLINE Complex<Type>
operator-(const Complex<Type>& lhs, const Complex<Type>& rhs)
{
    return Complex<Type>(lhs.real - rhs.real, lhs.imaginary - rhs.imaginary);
}

template<typename Type>
HDINLINE Complex<Type>
operator-(const Complex<Type>& lhs, const Type& rhs)
{
    return Complex<Type>(lhs.real - rhs, lhs.imaginary);
}

template<typename Type>
HDINLINE Complex<Type>
operator-(const Type& lhs, const Complex<Type>& rhs)
{
    return Complex<Type>(lhs - rhs.real, -rhs.imaginary);
}

/** Multiplication operators */

template<typename Type>
HDINLINE Complex<Type>
operator*(const Complex<Type>& lhs, const Complex<Type>& rhs)
{
    return Complex<Type>(lhs.real * rhs.real - lhs.imaginary * rhs.imaginary,
                     lhs.imaginary * rhs.real + lhs.real * rhs.imaginary);
}

template<typename Type>
HDINLINE Complex<Type>
operator*(const Complex<Type>& lhs, const Type& rhs)
{
    return Complex<Type>(lhs.real * rhs, lhs.imaginary * rhs);
}

template<typename Type>
HDINLINE Complex<Type>
operator*(const Type& lhs, const Complex<Type>& rhs)
{
    return Complex<Type>(lhs * rhs.real, lhs * rhs.imaginary);
}

/** Division operators */

template<typename Type>
HDINLINE Complex<Type>
operator/(const Complex<Type>& lhs, const Type& rhs)
{
    return Complex<Type>(lhs.real / rhs, lhs.imaginary / rhs);
}

template<typename Type>
HDINLINE Complex<Type>
operator/(const Type& lhs, const Complex<Type>& rhs)
{
    return Complex<Type>(lhs * rhs.real/(rhs.real*rhs.real+rhs.imaginary*rhs.imaginary),
                     -lhs * rhs.imaginary/( rhs.real*rhs.real+rhs.imaginary*rhs.imaginary ));
}

template<typename Type>
HDINLINE Complex<Type>
operator/(const Complex<Type>& lhs, const Complex<Type>& rhs)
{
    return lhs*Complex<Type>(rhs.real/(rhs.real*rhs.real+rhs.imaginary*rhs.imaginary),
                        -rhs.imaginary/( rhs.real*rhs.real+rhs.imaginary*rhs.imaginary ));
}

} //namespace math
} //namespace PMacc
