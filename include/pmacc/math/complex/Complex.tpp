/* Copyright 2013-2019 Heiko Burau, Rene Widera, Richard Pausch,
 *                     Alexander Debus, Benjamin Worpitz, Finn-Ole Carstens
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

#include "pmacc/algorithms/math.hpp"
#include "pmacc/algorithms/TypeCast.hpp"
#include "pmacc/math/Complex.hpp"

#include "pmacc/traits/GetComponentsType.hpp"
#include "pmacc/traits/GetNComponents.hpp"

#include <cmath>

namespace pmacc
{
namespace algorithms
{
namespace math
{

namespace pmMath = pmacc::algorithms::math;

/*  Set primary template and subsequent specialization for returning a complex number
    by using Euler's formula. */

template<typename T_Type>
struct Euler;

template<typename T_Type>
HDINLINE typename Euler< T_Type >::result euler(const T_Type& magnitude, const T_Type& phase)
{
    return Euler< T_Type > ()(magnitude, phase);
}

template<typename T_Type>
HDINLINE typename Euler< T_Type >::result euler(const T_Type& magnitude, const T_Type& sinValue,
                                                const T_Type& cosValue)
{
    return Euler< T_Type > ()(magnitude, sinValue, cosValue);
}

template<typename T_Type>
struct Euler
{
    typedef typename ::pmacc::math::Complex<T_Type> result;

    HDINLINE result operator( )(const T_Type &magnitude, const T_Type &phase)
    {
        return result(magnitude * pmMath::cos(phase),magnitude * pmMath::sin(phase));
    }

    HDINLINE result operator( )(const T_Type &magnitude,
                                const T_Type &sinValue, const T_Type &cosValue)
    {
        return result(magnitude * cosValue, magnitude * sinValue);
    }
};

/* Specialize sqrt() for complex numbers. */

template<typename T_Type>
struct Sqrt< ::pmacc::math::Complex<T_Type> >
{
    typedef typename ::pmacc::math::Complex<T_Type> result;
    typedef T_Type type;

    HDINLINE result operator( )(const ::pmacc::math::Complex<T_Type>& other)
    {
        if (other.get_real()<=type(0.0) && other.get_imag()==type(0.0) ) {
            return ::pmacc::math::Complex<T_Type>(type(0.0), pmMath::sqrt( -other.get_real() ) );
        }
        else {
            return pmMath::sqrt( pmMath::abs(other) )*(other+pmMath::abs(other))
                /pmMath::abs(other+pmMath::abs(other));
        }
    }
};

/* Specialize exp() for complex numbers. */

template<typename T_Type>
struct Exp< ::pmacc::math::Complex<T_Type> >
{
    typedef typename ::pmacc::math::Complex<T_Type> result;
    typedef T_Type type;

    HDINLINE result operator( )(const ::pmacc::math::Complex<T_Type>& other)
    {
        return pmMath::euler(type(1.0),other.get_imag())*pmMath::exp(other.get_real());
    }
};

/*  Set primary template and subsequent specialization of arg() for retrieving
 *  the phase of a complex number (Note: Branchcut running from -infinity to 0).
 */
template<typename T_Type>
struct Arg;

template<typename T_Type>
HDINLINE typename Arg< T_Type >::result arg(const T_Type& val)
{
    return Arg< T_Type > ()(val);
}

template<typename T_Type>
struct Arg< ::pmacc::math::Complex<T_Type> >
{
    typedef typename ::pmacc::math::Complex<T_Type>::type result;
    typedef T_Type type;

    HDINLINE result operator( )(const ::pmacc::math::Complex<T_Type>& other)
    {
        if ( other.get_real()==type(0.0) && other.get_imag()==type(0.0) )
            return type(0.0);
        else if ( other.get_real()==type(0.0) && other.get_imag()>type(0.0) )
            return Pi< type >::halfValue;
        else if ( other.get_real()==type(0.0) && other.get_imag()<type(0.0) )
            return -Pi< type >::halfValue;
        else if ( other.get_real()<type(0.0) && other.get_imag()==type(0.0) )
            return Pi< type >::value;
        else
            return pmMath::atan2(other.get_imag(),other.get_real());
    }
};

/*  Specialize pow() for complex numbers. */
template<typename T_Type>
struct Pow< ::pmacc::math::Complex<T_Type>, T_Type >
{
    typedef typename ::pmacc::math::Complex<T_Type> result;
    typedef T_Type type;

    HDINLINE result operator( )(const ::pmacc::math::Complex<T_Type>& other,
                                const T_Type& exponent)
    {
        return pmMath::pow( pmMath::abs(other),exponent )
                *pmMath::exp( ::pmacc::math::Complex<T_Type>(type(0.),type(1.) )
                *pmMath::arg(other)*exponent );
    }
};

/*  Specialize abs() for complex numbers. */
template<typename T_Type>
struct Abs< ::pmacc::math::Complex<T_Type> >
{
    typedef typename ::pmacc::math::Complex<T_Type>::type result;

    HDINLINE result operator( )(const ::pmacc::math::Complex<T_Type>& other)
    {
        return pmMath::sqrt( pmMath::abs2(other.get_real()) + pmMath::abs2(other.get_imag()) );
    }
};

/*  Specialize abs2() for complex numbers. */
template<typename T_Type>
struct Abs2< ::pmacc::math::Complex<T_Type> >
{
    typedef typename ::pmacc::math::Complex<T_Type>::type result;

    HDINLINE result operator( )(const ::pmacc::math::Complex<T_Type>& other)
    {
        return pmMath::abs2(other.get_real()) + pmMath::abs2(other.get_imag());
    }
};

    /*  Specialize log() for complex numbers. */
    template< typename T_Type >
    struct Log< ::pmacc::math::Complex< T_Type > >
    {
        using type = T_Type;
        using result = typename ::pmacc::math::Complex< type >::type;

        HDINLINE result operator( )( ::pmacc::math::Complex< T_Type > const & other )
        {
            return pmMath::log( pmMath::abs( other ) ) +
                ::pmacc::math::Complex< T_Type >(
                    type( 0. ),
                    type( 1. )
                ) * pmMath::arg( other );
        }
    };

    /*  Specialize sin( ) for complex numbers. */
    template< typename T_Type >
    struct Sin< ::pmacc::math::Complex< T_Type > >
    {
        using result = typename ::pmacc::math::Complex< T_Type >;
        using type = T_Type;

        HDINLINE result operator( )( const ::pmacc::math::Complex< T_Type > & other )
        {
            return ( pmMath::exp( ::pmacc::math::Complex< T_Type >( type( 0. ), type( 1. ) ) * other ) -
                   pmMath::exp( ::pmacc::math::Complex< T_Type >( type( 0. ), type( -1. ) ) * other ) ) /
                   ::pmacc::math::Complex< T_Type >( type( 0. ), type( 2. ) );
        }
    };

    /*  Specialize cos( ) for complex numbers. */
    template< typename T_Type >
    struct Cos< ::pmacc::math::Complex< T_Type > >
    {
        using result = typename ::pmacc::math::Complex< T_Type >;
        using type = T_Type;

        HDINLINE result operator( )( const ::pmacc::math::Complex< T_Type >& other )
        {
            return ( pmMath::exp( ::pmacc::math::Complex< T_Type >( type( 0. ), type( 1. ) ) * other ) +
                   pmMath::exp( ::pmacc::math::Complex< T_Type >( type( 0. ), type( -1. ) ) * other ) ) /
                   type( 2.0 );
        }
    };

} //namespace math
} //namespace algorithms
} //namespace pmacc

namespace pmacc
{
namespace algorithms
{
namespace precisionCast
{

/*  Specialize precisionCast-operators for complex numbers. */

template<typename T_CastToType>
struct TypeCast<T_CastToType, ::pmacc::math::Complex<T_CastToType> >
{
    typedef const ::pmacc::math::Complex<T_CastToType>& result;

    HDINLINE result operator( )(const ::pmacc::math::Complex<T_CastToType>& complexNumber ) const
    {
        return complexNumber;
    }
};

template<typename T_CastToType, typename T_OldType>
struct TypeCast<T_CastToType, ::pmacc::math::Complex<T_OldType> >
{
    typedef ::pmacc::math::Complex<T_CastToType> result;

    HDINLINE result operator( )(const ::pmacc::math::Complex<T_OldType>& complexNumber ) const
    {
        return result( complexNumber );
    }
};

} //namespace typecast
} //namespace algorithms

namespace mpi
{

    using complex_X = pmacc::math::Complex< picongpu::float_X >;

    // Specialize complex type grid buffer for MPI
    template<>
    MPI_StructAsArray getMPI_StructAsArray< pmacc::math::Complex<picongpu::float_X> >()
    {
        MPI_StructAsArray result = getMPI_StructAsArray< complex_X::type > ();
        result.sizeMultiplier *= uint32_t(sizeof(complex_X) / sizeof(typename complex_X::type));
        return result;
    };

} //namespace mpi
} //namespace pmacc
