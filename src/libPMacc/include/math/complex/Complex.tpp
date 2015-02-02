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

#include <cmath>
#include "math/Complex.hpp"
#include "algorithms/math.hpp"
#include "algorithms/TypeCast.hpp"
 
namespace PMacc
{
namespace algorithms
{
namespace math
{

/* specialize setting complex number by using Euler's formula */

template<typename Type>
struct Euler;

template<typename Type>
HDINLINE typename Euler< Type >::result euler(const Type& magnitude, const Type& phase)
{
    return Euler< Type > ()(magnitude, phase);
}

template<typename Type>
HDINLINE typename Euler< Type >::result euler(const Type& magnitude, const Type& sinValue, const Type& cosValue)
{
    return Euler< Type > ()(magnitude, sinValue, cosValue);
}

template<typename Type>
struct Euler
{
    typedef typename ::PMacc::math::Complex<Type> result;
    
    HDINLINE result operator( )(const Type &magnitude, const Type &phase)
    {
        return result(magnitude * picongpu::math::cos(phase),magnitude * picongpu::math::sin(phase));
    }
    
    HDINLINE result operator( )(const Type &magnitude, const Type &sinValue, const Type &cosValue)
    {
        return result(magnitude * cosValue, magnitude * sinValue);
    }
};

template<typename Type>
struct Sqrt< ::PMacc::math::Complex<Type> >
{
    typedef typename ::PMacc::math::Complex<Type> result;
    
    HDINLINE result operator( )(const ::PMacc::math::Complex<Type>& other)
    {
        if (other.get_real()<=0.0 && other.get_imag()==0.0) {
            return ::PMacc::math::Complex<Type>(0.0, PMacc::algorithms::math::sqrt( -other.get_real() ) );
        }
        else {
            return PMacc::algorithms::math::sqrt( PMacc::algorithms::math::abs(other) )*(other+PMacc::algorithms::math::abs(other))
                /PMacc::algorithms::math::abs(other+PMacc::algorithms::math::abs(other));
        }
    }
};

template<typename Type>
struct Exp< ::PMacc::math::Complex<Type> >
{
    typedef typename ::PMacc::math::Complex<Type> result;
    
    HDINLINE result operator( )(const ::PMacc::math::Complex<Type>& other)
    {
        return PMacc::algorithms::math::euler(1.0,other.get_imag())*PMacc::algorithms::math::exp(other.get_real());
    }
};

// Phase of complex number (Note: Branchcut running from -infinity to 0)

template<typename Type>
struct Arg;

template<typename Type>
HDINLINE typename Arg< Type >::result arg(const Type& val)
{
    return Arg< Type > ()(val);
}

template<typename Type>
struct Arg< ::PMacc::math::Complex<Type> >
{
    typedef typename ::PMacc::math::Complex<Type>::type result;
    
    HDINLINE result operator( )(const ::PMacc::math::Complex<Type>& other)
    {
        if (other.get_real()==0.0 && other.get_imag()==0.0) return 0.0;
        else if (other.get_real()==0.0 && other.get_imag()>0.0) return M_PI/2;
        else if (other.get_real()==0.0 && other.get_imag()<0.0) return -M_PI/2;
        else if (other.get_real()<0.0 && other.get_imag()==0.0) return M_PI;
        else return PMacc::algorithms::math::atan2(other.get_imag(),other.get_real());
    }
};

template<typename Type>
struct Pow< ::PMacc::math::Complex<Type>, Type >
{
    typedef typename ::PMacc::math::Complex<Type> result;
    
    HDINLINE result operator( )(const ::PMacc::math::Complex<Type>& other, const Type& exponent)
    {
        return PMacc::algorithms::math::pow( PMacc::algorithms::math::abs(other),exponent )
                *PMacc::algorithms::math::exp( ::PMacc::math::Complex<Type>(0.,1.)*PMacc::algorithms::math::arg(other)*exponent );
    }
};

template<typename Type>
struct Abs< ::PMacc::math::Complex<Type> >
{
    typedef typename ::PMacc::math::Complex<Type>::type result;

    HDINLINE result operator( )(const ::PMacc::math::Complex<Type>& other)
    {
        return PMacc::algorithms::math::sqrt( PMacc::algorithms::math::abs2(other.get_real()) + PMacc::algorithms::math::abs2(other.get_imag()) );
    }
};

template<typename Type>
struct Abs2< ::PMacc::math::Complex<Type> >
{
    typedef typename ::PMacc::math::Complex<Type>::type result;
    
    HDINLINE result operator( )(const ::PMacc::math::Complex<Type>& other)
    {
        return PMacc::algorithms::math::abs2(other.get_real()) + PMacc::algorithms::math::abs2(other.get_imag());
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

template<typename CastToType>
struct TypeCast<CastToType, ::PMacc::math::Complex<CastToType> >
{
    typedef const ::PMacc::math::Complex<CastToType>& result;

    HDINLINE result operator( )(const ::PMacc::math::Complex<CastToType>& complexNumber ) const
    {
        return complexNumber;
    }
};

template<typename CastToType, typename OldType>
struct TypeCast<CastToType, ::PMacc::math::Complex<OldType> >
{
    typedef ::PMacc::math::Complex<CastToType> result;

    HDINLINE result operator( )(const ::PMacc::math::Complex<OldType>& complexNumber ) const
    {
        return result( complexNumber );
    }
};

} //namespace typecast
} //namespace algorithms
} //PMacc
