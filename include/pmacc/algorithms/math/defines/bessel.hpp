/* Copyright 2016-2017 Alexander Debus
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

namespace pmacc
{
namespace algorithms
{
namespace math
{

    /* Modified cylindrical Bessel function of order 1 */

    template< typename T_Type >
    struct Cyl_bessel_i1;

    /**
     * Calculate the value of the regular modified cylindrical Bessel function
     * of order 1 for the input argument.
     * @return float value
     */
    template<typename T_Type >
    HDINLINE typename Cyl_bessel_i1<T_Type>::result cyl_bessel_i1( T_Type x )
    {
        return Cyl_bessel_i1< T_Type >( )( x );
    }

    /* Modified cylindrical Bessel function of order 0 */

    template< typename T_Type >
    struct Cyl_bessel_i0;

    /**
     * Calculate the value of the regular modified cylindrical Bessel function
     * of order 0 for the input argument.
     * @return float value
     */
    template< typename T_Type >
    HDINLINE typename Cyl_bessel_i0< T_Type >::result cyl_bessel_i0( T_Type x )
    {
        return Cyl_bessel_i0< T_Type >( )( x );
    }

    /* Bessel function of first kind of order 0 */

    template< typename T_Type >
    struct J0;

    /**
     * Calculate the value of the Bessel function
     * of first kind of order 0 for the input argument.
     * @return float value
     */
    template< typename T_Type >
    HDINLINE typename J0< T_Type >::result j0( T_Type x )
    {
        return J0< T_Type >( )( x );
    }

    /* Bessel function of first kind of order 1 */

    template< typename T_Type >
    struct J1;

    /**
     * Calculate the value of the Bessel function
     * of first kind of order 1 for the input argument.
     * @return float value
     */
    template< typename T_Type >
    HDINLINE typename J1< T_Type >::result j1( T_Type x )
    {
        return J1< T_Type >( )( x );
    }

    /* Bessel function of first kind of order n */

    template<
        typename T_IntType,
        typename T_FloatType
    >
    struct Jn;

    /**
     * Calculate the value of the Bessel function
     * of first kind of order n for the input argument.
     * @param n nth order
     * @param x input argument
     * @return float value
     */
    template<
        typename T_IntType,
        typename T_FloatType
    >
    HDINLINE
    typename Jn<
        T_IntType,
        T_FloatType
    >::result
    jn(
        T_IntType const & n,
        T_FloatType const & x
    )
    {
        return Jn<
            T_IntType,
            T_FloatType
        >( )(
            n,
            x
        );
    }

    /* Bessel function of second kind of order 0 */

    template< typename T_Type >
    struct Y0;

    /**
     * Calculate the value of the Bessel function
     * of first kind of order 0 for the input argument.
     * @return float value
     */
    template< typename T_Type >
    HDINLINE typename Y0< T_Type >::result y0( T_Type x )
    {
        return Y0< T_Type >( )( x );
    }

    /* Bessel function of second kind of order 1 */

    template< typename T_Type >
    struct Y1;

    /**
     * Calculate the value of the Bessel function
     * of second kind of order 1 for the input argument.
     * @return float value
     */
    template< typename T_Type >
    HDINLINE typename Y1< T_Type >::result y1( T_Type x )
    {
        return Y1< T_Type >( )( x );
    }

    /** Bessel function of second kind of order n */

    template<
        typename T_IntType,
        typename T_FloatType
    >
    struct Yn;

    /** Bessel function of second kind of order n
     *
     * Calculate the value of the Bessel function
     * of second kind of order n for the input argument.
     *
     * @param n nth order
     * @param x input argument
     * @return float value
     */
    template<
        typename T_IntType,
        typename T_FloatType
    >
    HDINLINE
    typename Yn<
        T_IntType,
        T_FloatType
    >::result
    yn(
        T_IntType const & n,
        T_FloatType const & x
    )
    {
        return Yn<
            T_IntType,
            T_FloatType
        >( )(
          n,
          x
        );
    }


} //namespace math
} //namespace algorithms
} //namespace pmacc
