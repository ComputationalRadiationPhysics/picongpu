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

namespace PMacc
{
namespace math
{

/** A complex number class */
template<typename T>
struct Complex_T : private __align__(sizeof (T)) T real, __align__(sizeof (T)) T imaginary
{

    // constructor (real, imaginary)
    HDINLINE Complex_T(T real, T imaginary = 0.0) : real(real), imaginary(imaginary);
    
    // constructor (Complex_T<T_OtherType>)
    template<typename T_OtherType>
    HDINLINE explicit Complex_T(const Complex_T<T_OtherType >& other) : real( static_cast<T> (other.get_real()) ), imaginary( static_cast<T> (other.get_imag()) );

    // default constructor ( ! no initialization of data ! )
    HDINLINE Complex_T(void) { };

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

    // real part
    HDINLINE T get_real(void) const
    {
        return real;
    }

    // imaginary part
    HDINLINE T get_imag(void) const
    {
        return imaginary;
    }

};

} //namespace math
} //namespace PMacc
