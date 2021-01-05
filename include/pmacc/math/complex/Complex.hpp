/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch, Alexander Debus
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include "pmacc/types.hpp"

namespace pmacc
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
            HDINLINE Complex(T_Type real, T_Type imaginary = type(0.0)) : real(real), imaginary(imaginary)
            {
            }

            constexpr HDINLINE Complex(const Complex& other) = default;

            // constructor (Complex<T_OtherType>)
            template<typename T_OtherType>
            HDINLINE explicit Complex(const Complex<T_OtherType>& other)
                : real(static_cast<T_Type>(other.get_real()))
                , imaginary(static_cast<T_Type>(other.get_imag()))
            {
            }

            // default constructor ( ! no initialization of data ! )
            HDINLINE Complex(void)
            {
            }

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
                real = other.get_real();
                imaginary = other.get_imag();
                return *this;
            }

            // assign addition
            HDINLINE Complex& operator+=(const Complex& other)
            {
                real += other.get_real();
                imaginary += other.get_imag();
                return *this;
            }

            // assign difference
            HDINLINE Complex& operator-=(const Complex& other)
            {
                real -= other.get_real();
                imaginary -= other.get_imag();
                return *this;
            }

            // assign multiplication
            HDINLINE Complex& operator*=(const Complex& other)
            {
                *this = *this * other;
                return *this;
            }

            // real part
            HDINLINE T_Type& get_real()
            {
                return real;
            }

            // real part
            HDINLINE T_Type get_real(void) const
            {
                return real;
            }

            // imaginary part
            HDINLINE T_Type& get_imag()
            {
                return imaginary;
            }

            // imaginary part
            HDINLINE T_Type get_imag(void) const
            {
                return imaginary;
            }

            // complex zero
            HDINLINE static Complex<T_Type> zero(void)
            {
                return Complex<T_Type>(type(0.0), type(0.0));
            }

        private:
            PMACC_ALIGN(real, T_Type); // real part
            PMACC_ALIGN(imaginary, T_Type); // imaginary part
        };

        /** Addition operators */

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator+(const Complex<T_Type>& lhs, const Complex<T_Type>& rhs)
        {
            return Complex<T_Type>(lhs.get_real() + rhs.get_real(), lhs.get_imag() + rhs.get_imag());
        }

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator+(const Complex<T_Type>& lhs, const T_Type& rhs)
        {
            return Complex<T_Type>(lhs.get_real() + rhs, lhs.get_imag());
        }

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator+(const T_Type& lhs, const Complex<T_Type>& rhs)
        {
            return Complex<T_Type>(lhs + rhs.get_real(), rhs.get_imag());
        }

        /** Subtraction operators */

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator-(const Complex<T_Type>& lhs, const Complex<T_Type>& rhs)
        {
            return Complex<T_Type>(lhs.get_real() - rhs.get_real(), lhs.get_imag() - rhs.get_imag());
        }

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator-(const Complex<T_Type>& lhs, const T_Type& rhs)
        {
            return Complex<T_Type>(lhs.get_real() - rhs, lhs.get_imag());
        }

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator-(const T_Type& lhs, const Complex<T_Type>& rhs)
        {
            return Complex<T_Type>(lhs - rhs.get_real(), -rhs.get_imag());
        }

        /** Multiplication operators */

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator*(const Complex<T_Type>& lhs, const Complex<T_Type>& rhs)
        {
            return Complex<T_Type>(
                lhs.get_real() * rhs.get_real() - lhs.get_imag() * rhs.get_imag(),
                lhs.get_imag() * rhs.get_real() + lhs.get_real() * rhs.get_imag());
        }

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator*(const Complex<T_Type>& lhs, const T_Type& rhs)
        {
            return Complex<T_Type>(lhs.get_real() * rhs, lhs.get_imag() * rhs);
        }

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator*(const T_Type& lhs, const Complex<T_Type>& rhs)
        {
            return Complex<T_Type>(lhs * rhs.get_real(), lhs * rhs.get_imag());
        }

        /** Division operators */

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator/(const Complex<T_Type>& lhs, const T_Type& rhs)
        {
            return Complex<T_Type>(lhs.get_real() / rhs, lhs.get_imag() / rhs);
        }

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator/(const T_Type& lhs, const Complex<T_Type>& rhs)
        {
            return Complex<T_Type>(
                lhs * rhs.get_real() / (rhs.get_real() * rhs.get_real() + rhs.get_imag() * rhs.get_imag()),
                -lhs * rhs.get_imag() / (rhs.get_real() * rhs.get_real() + rhs.get_imag() * rhs.get_imag()));
        }

        template<typename T_Type>
        HDINLINE Complex<T_Type> operator/(const Complex<T_Type>& lhs, const Complex<T_Type>& rhs)
        {
            return lhs
                * Complex<T_Type>(
                       rhs.get_real() / (rhs.get_real() * rhs.get_real() + rhs.get_imag() * rhs.get_imag()),
                       -rhs.get_imag() / (rhs.get_real() * rhs.get_real() + rhs.get_imag() * rhs.get_imag()));
        }

    } // namespace math
} // namespace pmacc
