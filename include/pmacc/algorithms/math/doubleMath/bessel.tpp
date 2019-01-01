/* Copyright 2016-2019 Alexander Debus
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
namespace algorithms
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
#if __CUDA_ARCH__
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
#if __CUDA_ARCH__
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
#if __CUDA_ARCH__
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
#if __CUDA_ARCH__
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
#if __CUDA_ARCH__
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
#if __CUDA_ARCH__
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
#if __CUDA_ARCH__
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
#if __CUDA_ARCH__
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
} //namespace algorithms
} //namespace pmacc
