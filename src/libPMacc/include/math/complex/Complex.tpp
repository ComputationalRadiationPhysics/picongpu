/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera, Richard Pausch,
 *                     Alexander Debus, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
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

#include "algorithms/math.hpp"
#include "algorithms/TypeCast.hpp"
#include "math/Complex.hpp"

#include <cmath>

namespace PMacc
{
namespace algorithms
{
namespace math
{

namespace pmMath = PMacc::algorithms::math;

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
    typedef typename ::PMacc::math::Complex<T_Type> result;

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
struct Sqrt< ::PMacc::math::Complex<T_Type> >
{
    typedef typename ::PMacc::math::Complex<T_Type> result;
    typedef T_Type type;

    HDINLINE result operator( )(const ::PMacc::math::Complex<T_Type>& other)
    {
        if (other.get_real()<=type(0.0) && other.get_imag()==type(0.0) ) {
            return ::PMacc::math::Complex<T_Type>(type(0.0), pmMath::sqrt( -other.get_real() ) );
        }
        else {
            return pmMath::sqrt( pmMath::abs(other) )*(other+pmMath::abs(other))
                /pmMath::abs(other+pmMath::abs(other));
        }
    }
};

/* Specialize exp() for complex numbers. */

template<typename T_Type>
struct Exp< ::PMacc::math::Complex<T_Type> >
{
    typedef typename ::PMacc::math::Complex<T_Type> result;
    typedef T_Type type;

    HDINLINE result operator( )(const ::PMacc::math::Complex<T_Type>& other)
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
struct Arg< ::PMacc::math::Complex<T_Type> >
{
    typedef typename ::PMacc::math::Complex<T_Type>::type result;
    typedef T_Type type;

    HDINLINE result operator( )(const ::PMacc::math::Complex<T_Type>& other)
    {
        if ( other.get_real()==type(0.0) && other.get_imag()==type(0.0) )
            return type(0.0);
        else if ( other.get_real()==type(0.0) && other.get_imag()>type(0.0) )
            return type(M_PI)/type(2.0);
        else if ( other.get_real()==type(0.0) && other.get_imag()<type(0.0) )
            return type(-M_PI)/type(2.0);
        else if ( other.get_real()<type(0.0) && other.get_imag()==type(0.0) )
            return type(M_PI);
        else
            return pmMath::atan2(other.get_imag(),other.get_real());
    }
};

/*  Specialize pow() for complex numbers. */
template<typename T_Type>
struct Pow< ::PMacc::math::Complex<T_Type>, T_Type >
{
    typedef typename ::PMacc::math::Complex<T_Type> result;
    typedef T_Type type;

    HDINLINE result operator( )(const ::PMacc::math::Complex<T_Type>& other,
                                const T_Type& exponent)
    {
        return pmMath::pow( pmMath::abs(other),exponent )
                *pmMath::exp( ::PMacc::math::Complex<T_Type>(type(0.),type(1.) )
                *pmMath::arg(other)*exponent );
    }
};

/*  Specialize abs() for complex numbers. */
template<typename T_Type>
struct Abs< ::PMacc::math::Complex<T_Type> >
{
    typedef typename ::PMacc::math::Complex<T_Type>::type result;

    HDINLINE result operator( )(const ::PMacc::math::Complex<T_Type>& other)
    {
        return pmMath::sqrt( pmMath::abs2(other.get_real()) + pmMath::abs2(other.get_imag()) );
    }
};

/*  Specialize abs2() for complex numbers. */
template<typename T_Type>
struct Abs2< ::PMacc::math::Complex<T_Type> >
{
    typedef typename ::PMacc::math::Complex<T_Type>::type result;

    HDINLINE result operator( )(const ::PMacc::math::Complex<T_Type>& other)
    {
        return pmMath::abs2(other.get_real()) + pmMath::abs2(other.get_imag());
    }
};

} //namespace math
} //namespace algorithms
} // namespace PMacc

namespace PMacc
{
namespace algorithms
{
namespace precisionCast
{

/*  Specialize precisionCast-operators for complex numbers. */

template<typename T_CastToType>
struct TypeCast<T_CastToType, ::PMacc::math::Complex<T_CastToType> >
{
    typedef const ::PMacc::math::Complex<T_CastToType>& result;

    HDINLINE result operator( )(const ::PMacc::math::Complex<T_CastToType>& complexNumber ) const
    {
        return complexNumber;
    }
};

template<typename T_CastToType, typename T_OldType>
struct TypeCast<T_CastToType, ::PMacc::math::Complex<T_OldType> >
{
    typedef ::PMacc::math::Complex<T_CastToType> result;

    HDINLINE result operator( )(const ::PMacc::math::Complex<T_OldType>& complexNumber ) const
    {
        return result( complexNumber );
    }
};

} //namespace typecast
} //namespace algorithms
} //PMacc
