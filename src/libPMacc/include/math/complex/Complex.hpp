/**
 * 2013-2015 Heiko Burau, Rene Widera, Richard Pausch, Alexander Debus
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
template<typename T_Type>
struct Complex
{

public:

    typedef T_Type type;

    // constructor (real, imaginary)
    HDINLINE Complex(T_Type real, T_Type imaginary = type(0.0) ) : real(real), imaginary(imaginary) { }
    
    // constructor (Complex<T_OtherType>)
    template<typename T_OtherType>
    HDINLINE explicit Complex(const Complex<T_OtherType >& other) : 
                        real( static_cast<T_Type> (other.get_real()) ),
                        imaginary( static_cast<T_Type> (other.get_imag()) ) { }

    // default constructor ( ! no initialization of data ! )
    HDINLINE Complex(void) { }

    // Conversion from scalar (assignment)
    HDINLINE Complex& operator=(const T_Type& other)
    {
        real = other;
        imaginary = type(0.0);
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
    HDINLINE T_Type get_real(void) const
    {
        return real;
    }

    // imaginary part
    HDINLINE T_Type get_imag(void) const
    {
        return imaginary;
    }
    
    // complex zero
    HDINLINE static Complex<T_Type> zero(void)
    {
        return Complex<T_Type>( type(0.0) , type(0.0) );
    }
    
private:
    PMACC_ALIGN(real,T_Type); // real part
    PMACC_ALIGN(imaginary,T_Type); // imaginary part
    
};

/** Addition operators */

template<typename T_Type>
HDINLINE Complex<T_Type>
operator+(const Complex<T_Type>& lhs, const Complex<T_Type>& rhs)
{
    return Complex<T_Type>(lhs.real + rhs.real, lhs.imaginary + rhs.imaginary);
}

template<typename T_Type>
HDINLINE Complex<T_Type>
operator+(const Complex<T_Type>& lhs, const T_Type& rhs)
{
    return Complex<T_Type>(lhs.real + rhs, lhs.imaginary);
}

template<typename T_Type>
HDINLINE Complex<T_Type>
operator+(const T_Type& lhs, const Complex<T_Type>& rhs)
{
    return Complex<T_Type>(lhs + rhs.real, rhs.imaginary);
}

/** Substraction operators */

template<typename T_Type>
HDINLINE Complex<T_Type>
operator-(const Complex<T_Type>& lhs, const Complex<T_Type>& rhs)
{
    return Complex<T_Type>(lhs.real - rhs.real, lhs.imaginary - rhs.imaginary);
}

template<typename T_Type>
HDINLINE Complex<T_Type>
operator-(const Complex<T_Type>& lhs, const T_Type& rhs)
{
    return Complex<T_Type>(lhs.real - rhs, lhs.imaginary);
}

template<typename T_Type>
HDINLINE Complex<T_Type>
operator-(const T_Type& lhs, const Complex<T_Type>& rhs)
{
    return Complex<T_Type>(lhs - rhs.real, -rhs.imaginary);
}

/** Multiplication operators */

template<typename T_Type>
HDINLINE Complex<T_Type>
operator*(const Complex<T_Type>& lhs, const Complex<T_Type>& rhs)
{
    return Complex<T_Type>(lhs.real * rhs.real - lhs.imaginary * rhs.imaginary,
                     lhs.imaginary * rhs.real + lhs.real * rhs.imaginary);
}

template<typename T_Type>
HDINLINE Complex<T_Type>
operator*(const Complex<T_Type>& lhs, const T_Type& rhs)
{
    return Complex<T_Type>(lhs.real * rhs, lhs.imaginary * rhs);
}

template<typename T_Type>
HDINLINE Complex<T_Type>
operator*(const T_Type& lhs, const Complex<T_Type>& rhs)
{
    return Complex<T_Type>(lhs * rhs.real, lhs * rhs.imaginary);
}

/** Division operators */

template<typename T_Type>
HDINLINE Complex<T_Type>
operator/(const Complex<T_Type>& lhs, const T_Type& rhs)
{
    return Complex<T_Type>(lhs.real / rhs, lhs.imaginary / rhs);
}

template<typename T_Type>
HDINLINE Complex<T_Type>
operator/(const T_Type& lhs, const Complex<T_Type>& rhs)
{
    return Complex<T_Type>(lhs * rhs.real/(rhs.real*rhs.real+rhs.imaginary*rhs.imaginary),
                     -lhs * rhs.imaginary/( rhs.real*rhs.real+rhs.imaginary*rhs.imaginary ));
}

template<typename T_Type>
HDINLINE Complex<T_Type>
operator/(const Complex<T_Type>& lhs, const Complex<T_Type>& rhs)
{
    return lhs*Complex<T_Type>(rhs.real/(rhs.real*rhs.real+rhs.imaginary*rhs.imaginary),
                        -rhs.imaginary/( rhs.real*rhs.real+rhs.imaginary*rhs.imaginary ));
}

} //namespace math
} //namespace PMacc
