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

namespace util
{

    //goal: to increase readability of code

    template<typename A> /// a generic square function
    HDINLINE A square(A a)
    {
        return a*a;
    }

    template<typename A, typename R> /// a more generic square function
    HDINLINE R square(A a)
    {
        return a*a;
    }

    template<typename A> /// a generic cube function
    HDINLINE A cube(A a)
    {
        return a * a*a;
    }

    template<typename A, typename R> /// a more generic cube function
    HDINLINE R cube(A a)
    {
        return a * a*a;
    }

    template<typename A, typename R = A> /// a more generic square struct
            struct Cube
    {

        HDINLINE R operator()(A a)
        {
            return a * a*a;
        }
    };

    template<typename A, typename R = A> /// a more generic square struct
            struct Square
    {

        HDINLINE R operator()(A a) const
        {
            return a*a;
        }
    };

    /** compile time power function
      *
      * @tparam T - base type
      * @tparam Exp - Exponent
      * @return pow(x,Exp)=x^Exp
      */
    template< typename T, unsigned int Exp >
    struct Pow
    {
        HDINLINE T operator()( const T x ) const
        {
            Pow< T, Exp - 1 > pow;
            return x * pow( x );
        }
    };

    /** compile time power function
      *
      * specilization Exp = 1
      * @tparam T - base type
      * @return pow(x,1)=x
      */
    template< typename T >
    struct Pow< T, 1 >
    {
        HDINLINE T operator()( const T x ) const
        {
            return x;
        }
    };

    /** compile time power function
      *
      * specilization Exp = 0
      * (just as saveguard)
      * @tparam T - base type
      * @return pow(x,0)=1
      */
    template< typename T >
    struct Pow< T, 0 >
    {
        HDINLINE T operator()( const T x ) const
        {
            return 1;
        }
    };

} // namespace util
