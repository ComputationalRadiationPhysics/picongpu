/* Copyright 2016-2020 Alexander Debus
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
#include <boost/math/special_functions/bessel.hpp>


namespace pmacc
{
namespace math
{
namespace bessel
{

    template< >
    struct I0< double >
    {
        using result = double;

        HDINLINE result operator( )( result const & x )
        {
#if( CUPLA_DEVICE_COMPILE == 1) //we are on gpu
            return ::cyl_bessel_i0( x );
#else
            return boost::math::cyl_bessel_i(
                0,
                x
            );
#endif
        }
    };

    template< >
    struct I1< double >
    {
        using result = double;

        HDINLINE result operator( )( result const & x )
        {
#if( CUPLA_DEVICE_COMPILE == 1) //we are on gpu
            return ::cyl_bessel_i1( x );
#else
            return boost::math::cyl_bessel_i(
                1,
                x
            );
#endif
        }
    };

    template< >
    struct J0< double >
    {
        using result = double;

        HDINLINE result operator( )( result const & x )
        {
#if( defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__ == 1 ) //we are on gpu
            return ::j0( x );
#else
            return boost::math::cyl_bessel_j(
                0,
                x
            );
#endif
        }
    };

    template< >
    struct J1< double >
    {
        using result = double;

        HDINLINE result operator( )( result const & x )
        {
#if( defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__ == 1 ) //we are on gpu
            return ::j1( x );
#else
            return boost::math::cyl_bessel_j(
                1,
                x
            );
#endif
        }
    };

    template< >
    struct Jn<
        int,
        double
    >
    {
        using result = double;

        HDINLINE result operator( )(
            int const & n,
            result const & x
        )
        {
#if( defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__ == 1 ) //we are on gpu
            return ::jn(
                n,
                x
            );
#else
            return boost::math::cyl_bessel_j(
                n,
                x
            );
#endif
        }
    };

    template< >
    struct Y0< double >
    {
        using result = double;

        HDINLINE result operator( )( result const & x )
        {
#if( defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__ == 1 ) //we are on gpu
            return ::y0( x );
#else
            return boost::math::cyl_neumann(
                0,
                x
            );
#endif
        }
    };

    template< >
    struct Y1< double >
    {
        using result = double;

        HDINLINE result operator( )( result const & x )
        {
#if( defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__ == 1 ) //we are on gpu
            return ::y1( x );
#else
            return boost::math::cyl_neumann(
                1,
                x
            );
#endif
        }
    };

    template< >
    struct Yn<
        int,
        double
    >
    {
        using result = double;

        HDINLINE result operator( )(
            int const & n,
            result const & x
        )
        {
#if( defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__ == 1 ) //we are on gpu
            return ::yn(
                n,
                x
            );
#else
            return boost::math::cyl_neumann(
                n,
                x
            );
#endif
        }
    };

} //namespace bessel
} //namespace math
} //namespace pmacc
