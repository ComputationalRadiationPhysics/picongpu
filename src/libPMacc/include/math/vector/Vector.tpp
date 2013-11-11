/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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


#include "math/vector/Vector.hpp"
#include "algorithms/math.hpp"
#include "algorithms/TypeCast.hpp"
#include "mpi/GetMPI_StructAsArray.hpp"
#include "traits/GetComponentsType.hpp"
#include "traits/GetNComponents.hpp"

namespace PMacc
{
namespace traits
{

template<typename T_DataType, int T_Dim>
struct GetComponentsType<PMacc::math::Vector<T_DataType, T_Dim>, false >
{
    typedef typename PMacc::math::Vector<T_DataType, T_Dim>::type type;
};

template<typename T_DataType, int T_Dim>
struct GetNComponents<PMacc::math::Vector<T_DataType, T_Dim>,false >
{
    static const uint32_t value = (uint32_t) PMacc::math::Vector<T_DataType, T_Dim>::dim;
};

} //namespace traits
} //namespace PMacc


namespace PMacc
{
namespace algorithms
{
namespace math
{
namespace detail
{

/*#### comparison ############################################################*/

/*specialize max algorithm*/
template<typename Type, int dim>
struct Max< ::PMacc::math::Vector<Type, dim>, ::PMacc::math::Vector<Type, dim> >
{
    typedef ::PMacc::math::Vector<Type, dim> result;

    HDINLINE result operator( )(const ::PMacc::math::Vector<Type, dim> &vector1, const ::PMacc::math::Vector<Type, dim> &vector2 )
    {
        result tmp;
        for ( int i = 0; i < dim; ++i )
            tmp[i] = ::max( vector1[i], vector2[i] );
        return tmp;
    }
};

/*specialize max algorithm*/
template<typename Type, int dim>
struct Min< ::PMacc::math::Vector<Type, dim>, ::PMacc::math::Vector<Type, dim> >
{
    typedef ::PMacc::math::Vector<Type, dim> result;

    HDINLINE result operator( )(const ::PMacc::math::Vector<Type, dim> &vector1, const ::PMacc::math::Vector<Type, dim> &vector2 )
    {
        result tmp;
        for ( int i = 0; i < dim; ++i )
            tmp[i] = ::min( vector1[i], vector2[i] );
        return tmp;
    }
};

/*#### abs ###################################################################*/

/*specialize abs2 algorithm*/
template<typename Type, int dim>
struct Abs2< ::PMacc::math::Vector<Type, dim> >
{
    typedef typename ::PMacc::math::Vector<Type, dim>::type result;

    HDINLINE result operator( )(const ::PMacc::math::Vector<Type, dim> &vector )
    {
        result tmp = PMacc::algorithms::math::abs2( vector.x( ) );
        for ( int i = 1; i < dim; ++i )
            tmp += PMacc::algorithms::math::abs2( vector[i] );
        return tmp;
    }
};

/*specialize abs algorithm*/
template<typename Type, int dim>
struct Abs< ::PMacc::math::Vector<Type, dim> >
{
    typedef typename ::PMacc::math::Vector<Type, dim>::type result;

    HDINLINE result operator( )( ::PMacc::math::Vector<Type, dim> vector )
    {
        const result tmp = PMacc::algorithms::math::abs2( vector );
        return PMacc::algorithms::math::sqrt( tmp );
    }
};

/*#### cross #################################################################*/

template<typename Type>
struct Cross< ::PMacc::math::Vector<Type, DIM3>, ::PMacc::math::Vector<Type, DIM3> >
{
    typedef ::PMacc::math::Vector<Type, DIM3> myType;
    typedef myType result;

    HDINLINE myType operator( )(const myType& lhs, const myType & rhs )
    {
        return myType( lhs.y( ) * rhs.z( ) - lhs.z( ) * rhs.y( ),
                       lhs.z( ) * rhs.x( ) - lhs.x( ) * rhs.z( ),
                       lhs.x( ) * rhs.y( ) - lhs.y( ) * rhs.x( ) );
    }
};

/*#### dot ###################################################################*/

template<typename Type, int dim>
struct Dot< ::PMacc::math::Vector<Type, dim>, ::PMacc::math::Vector<Type, dim> >
{
    typedef ::PMacc::math::Vector<Type, dim> myType;
    typedef Type result;

    HDINLINE result operator( )(const myType& a, const myType & b )
    {
        BOOST_STATIC_ASSERT( dim > 0 );
        result tmp = a.x( ) * b.x( );
        for ( int i = 1; i < dim; i++ )
            tmp += a[i] * b[i];
        return tmp;
    }
};

/*#### pow ###################################################################*/

/*! Specialisation of pow where base is a vector and exponent is a scalar
 * 
 * Create pow separatley for every component of the vector.
 * 
 * @prama base vector with base values
 * @param exponent scalar with exponent value
 */
template<typename T1, typename T2, int dim>
struct Pow< ::PMacc::math::Vector<T1, dim>, T2 >
{
    typedef ::PMacc::math::Vector<T1, dim> Vector1;
    typedef Vector1 result;

    HDINLINE result operator( )(const Vector1& base, const T2 & exponent )
    {
        BOOST_STATIC_ASSERT( dim > 0 );
        result tmp;
        for ( int i = 0; i < dim; ++i )
            tmp[i] = PMacc::algorithms::math::pow( base[i], exponent );
        return tmp;
    }
};

}//namespace detail
} //namespace math
} //namespace algorithms
} // namespace PMacc

namespace PMacc
{
namespace algorithms
{
namespace typeCast
{
namespace detail
{

template<typename CastToType, int dim>
struct TypeCast<CastToType, ::PMacc::math::Vector<CastToType, dim> >
{
    typedef const ::PMacc::math::Vector<CastToType, dim>& result;

    HDINLINE result operator( )(const ::PMacc::math::Vector<CastToType, dim>& vector ) const
    {
        return vector;
    }
};

template<typename CastToType, typename OldType, int dim>
struct TypeCast<CastToType, ::PMacc::math::Vector<OldType, dim> >
{
    typedef ::PMacc::math::Vector<CastToType, dim> result;

    HDINLINE result operator( )(const ::PMacc::math::Vector<OldType, dim>& vector ) const
    {
        return result( vector );
    }
};
}//namespace detail
} //namespace typecast
} //namespace algorithms
} //PMacc

namespace PMacc
{
namespace algorithms
{
namespace promoteType
{

template<typename PromoteToType, typename OldType, int dim>
struct promoteType<PromoteToType, ::PMacc::math::Vector<OldType, dim> >
{
    typedef typename promoteType<OldType, PromoteToType>::ValueType PartType;
    typedef ::PMacc::math::Vector<PartType, dim> ValueType;
};

} //namespace promoteType
} //namespace algorithms
} //namespace PMacc

#include "mpi/GetMPI_StructAsArray.hpp"

namespace PMacc
{
namespace mpi
{
namespace def
{

template<int dim>
struct GetMPI_StructAsArray< ::PMacc::math::Vector<float, dim> >
{

    MPI_StructAsArray operator( )( ) const
    {
        return MPI_StructAsArray( MPI_FLOAT, dim );
    }
};

template<int dim>
struct GetMPI_StructAsArray< ::PMacc::math::Vector<double, dim> >
{

    MPI_StructAsArray operator( )( ) const
    {
        return MPI_StructAsArray( MPI_DOUBLE, dim );
    }
};

} //namespace def

} //namespace mpi

}//namespace PMacc
