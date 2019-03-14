/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
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


#include "pmacc/math/Vector.hpp"
#include "pmacc/algorithms/math.hpp"
#include "pmacc/algorithms/TypeCast.hpp"
#include "pmacc/algorithms/PromoteType.hpp"
#include "pmacc/mpi/GetMPI_StructAsArray.hpp"
#include "pmacc/traits/GetComponentsType.hpp"
#include "pmacc/traits/GetNComponents.hpp"
#include "pmacc/traits/GetInitializedInstance.hpp"

namespace pmacc
{
namespace traits
{

template<typename T_DataType, int T_Dim>
struct GetComponentsType<pmacc::math::Vector<T_DataType, T_Dim>, false >
{
    typedef typename pmacc::math::Vector<T_DataType, T_Dim>::type type;
};

template<typename T_DataType, int T_Dim>
struct GetNComponents<pmacc::math::Vector<T_DataType, T_Dim>,false >
{
    static constexpr uint32_t value = (uint32_t) pmacc::math::Vector<T_DataType, T_Dim>::dim;
};

template<typename T_Type, int T_dim, typename T_Accessor, typename T_Navigator, template<typename, int> class T_Storage>
struct GetInitializedInstance<math::Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage> >
{
    typedef math::Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage> Type;
    typedef typename Type::type ValueType;

    HDINLINE Type operator()(const ValueType value) const
    {
        return Type::create(value);
    }
};

} //namespace traits
} //namespace pmacc


namespace pmacc
{
namespace algorithms
{
namespace math
{

/*#### comparison ############################################################*/

/*specialize max algorithm*/
template<typename Type, int dim>
struct Max< ::pmacc::math::Vector<Type, dim>, ::pmacc::math::Vector<Type, dim> >
{
    typedef ::pmacc::math::Vector<Type, dim> result;

    HDINLINE result operator( )(const ::pmacc::math::Vector<Type, dim> &vector1, const ::pmacc::math::Vector<Type, dim> &vector2 )
    {
        result tmp;
        for ( int i = 0; i < dim; ++i )
            tmp[i] = pmacc::algorithms::math::max( vector1[i], vector2[i] );
        return tmp;
    }
};

/*specialize max algorithm*/
template<typename Type, int dim>
struct Min< ::pmacc::math::Vector<Type, dim>, ::pmacc::math::Vector<Type, dim> >
{
    typedef ::pmacc::math::Vector<Type, dim> result;

    HDINLINE result operator( )(const ::pmacc::math::Vector<Type, dim> &vector1, const ::pmacc::math::Vector<Type, dim> &vector2 )
    {
        result tmp;
        for ( int i = 0; i < dim; ++i )
            tmp[i] = pmacc::algorithms::math::min( vector1[i], vector2[i] );
        return tmp;
    }
};

/*#### abs ###################################################################*/

/*specialize abs2 algorithm*/
template<typename Type, int dim>
struct Abs2< ::pmacc::math::Vector<Type, dim> >
{
    typedef typename ::pmacc::math::Vector<Type, dim>::type result;

    HDINLINE result operator( )(const ::pmacc::math::Vector<Type, dim> &vector )
    {
        result tmp = pmacc::algorithms::math::abs2( vector.x( ) );
        for ( int i = 1; i < dim; ++i )
            tmp += pmacc::algorithms::math::abs2( vector[i] );
        return tmp;
    }
};

/*specialize abs algorithm*/
template<typename Type, int dim>
struct Abs< ::pmacc::math::Vector<Type, dim> >
{
    typedef typename ::pmacc::math::Vector<Type, dim>::type result;

    HDINLINE result operator( )( ::pmacc::math::Vector<Type, dim> vector )
    {
        const result tmp = pmacc::algorithms::math::abs2( vector );
        return pmacc::algorithms::math::sqrt( tmp );
    }
};

/*#### cross #################################################################*/

template<typename Type>
struct Cross< ::pmacc::math::Vector<Type, DIM3>, ::pmacc::math::Vector<Type, DIM3> >
{
    typedef ::pmacc::math::Vector<Type, DIM3> myType;
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
struct Dot< ::pmacc::math::Vector<Type, dim>, ::pmacc::math::Vector<Type, dim> >
{
    typedef ::pmacc::math::Vector<Type, dim> myType;
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
struct Pow< ::pmacc::math::Vector<T1, dim>, T2 >
{
    typedef ::pmacc::math::Vector<T1, dim> Vector1;
    typedef Vector1 result;

    HDINLINE result operator( )(const Vector1& base, const T2 & exponent )
    {
        BOOST_STATIC_ASSERT( dim > 0 );
        result tmp;
        for ( int i = 0; i < dim; ++i )
            tmp[i] = pmacc::algorithms::math::pow( base[i], exponent );
        return tmp;
    }
};

/*#### floor #################################################################*/

/*specialize floor algorithm*/
template<typename Type, int dim>
struct Floor< ::pmacc::math::Vector<Type, dim> >
{
    typedef ::pmacc::math::Vector<Type, dim> result;

    HDINLINE result operator( )( ::pmacc::math::Vector<Type, dim> &vector )
    {
        result tmp;
        for ( int i = 0; i < dim; ++i )
            tmp[i] = pmacc::algorithms::math::floor( vector[i] );
        return tmp;
    }
};


} //namespace math
} //namespace algorithms
} // namespace pmacc

namespace pmacc
{
namespace algorithms
{
namespace precisionCast
{

template<typename CastToType,
int dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage>
struct TypeCast<
    CastToType,
    ::pmacc::math::Vector<CastToType, dim, T_Accessor, T_Navigator, T_Storage>
>
{
    typedef const ::pmacc::math::Vector<
        CastToType,
        dim,
        T_Accessor,
        T_Navigator,
        T_Storage>& result;

    HDINLINE result operator( )( result vector ) const
    {
        return vector;
    }
};

template<typename CastToType,
typename OldType,
int dim,
typename T_Accessor,
typename T_Navigator,
template <typename, int> class T_Storage>
struct TypeCast<
    CastToType,
    ::pmacc::math::Vector<OldType, dim, T_Accessor, T_Navigator, T_Storage>
>
{
    typedef ::pmacc::math::Vector<CastToType, dim> result;
    typedef ::pmacc::math::Vector<OldType, dim, T_Accessor, T_Navigator, T_Storage> ParamType;

    HDINLINE result operator( )(const ParamType& vector ) const
    {
        return result( vector );
    }
};

} //namespace typecast
} //namespace algorithms
} //PMacc

namespace pmacc
{
namespace algorithms
{
namespace promoteType
{

template<typename PromoteToType, typename OldType, int dim>
struct promoteType<PromoteToType, ::pmacc::math::Vector<OldType, dim> >
{
    typedef typename promoteType<OldType, PromoteToType>::type PartType;
    typedef ::pmacc::math::Vector<PartType, dim> type;
};

} //namespace promoteType
} //namespace algorithms
} //namespace pmacc

namespace pmacc
{
namespace mpi
{
namespace def
{

template<int T_dim>
struct GetMPI_StructAsArray< ::pmacc::math::Vector<float, T_dim> >
{

    MPI_StructAsArray operator( )( ) const
    {
        return MPI_StructAsArray( MPI_FLOAT, T_dim );
    }
};

template<int T_dim, int T_N>
struct GetMPI_StructAsArray< ::pmacc::math::Vector<float, T_dim>[T_N] >
{

    MPI_StructAsArray operator( )( ) const
    {
        return MPI_StructAsArray( MPI_FLOAT, T_dim * T_N );
    }
};

template<int T_dim>
struct GetMPI_StructAsArray< ::pmacc::math::Vector<double, T_dim> >
{

    MPI_StructAsArray operator( )( ) const
    {
        return MPI_StructAsArray( MPI_DOUBLE, T_dim );
    }
};

template<int T_dim, int T_N>
struct GetMPI_StructAsArray< ::pmacc::math::Vector<double, T_dim>[T_N] >
{

    MPI_StructAsArray operator( )( ) const
    {
        return MPI_StructAsArray( MPI_DOUBLE, T_dim * T_N  );
    }
};

} //namespace def

} //namespace mpi

}//namespace pmacc
