/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera
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

#include <stdint.h>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/aux_/na.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/min_max.hpp>
#include <boost/mpl/times.hpp>
//#include <boost/mpl/arithmetic.hpp>
#include "../Vector.hpp"
#include <boost/mpl/int.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/accumulate.hpp>

namespace PMacc
{
namespace math
{
namespace CT
{

namespace mpl = boost::mpl;

namespace detail
{
template<int dim>
struct VectorFromCT;

template<>
struct VectorFromCT<1>
{
    template<typename Vec, typename CTVec>
    HDINLINE void operator()(Vec& vec, CTVec) const
    {
        BOOST_STATIC_ASSERT(Vec::dim == 1);
        BOOST_STATIC_ASSERT(CTVec::dim == 1);
        vec[0] = (typename Vec::type)CTVec::x::value;
    }
};
template<>
struct VectorFromCT<2>
{
    template<typename Vec, typename CTVec>
    HDINLINE void operator()(Vec& vec, CTVec) const
    {
        BOOST_STATIC_ASSERT(Vec::dim == 2);
        BOOST_STATIC_ASSERT(CTVec::dim == 2);
        vec[0] = (typename Vec::type)CTVec::x::value;
        vec[1] = (typename Vec::type)CTVec::y::value;
    }
};
template<>
struct VectorFromCT<3>
{
    template<typename Vec, typename CTVec>
    HDINLINE void operator()(Vec& vec, CTVec) const
    {
        BOOST_STATIC_ASSERT(Vec::dim == 3);
        BOOST_STATIC_ASSERT(CTVec::dim == 3);
        vec[0] = (typename Vec::type)CTVec::x::value;
        vec[1] = (typename Vec::type)CTVec::y::value;
        vec[2] = (typename Vec::type)CTVec::z::value;
    }
};

template<typename Arg0>
struct TypeSelector
{
    typedef typename Arg0::value_type type;
};
template<>
struct TypeSelector<mpl::na>
{
    typedef mpl::int_<0> type;
};

}

namespace mpl = boost::mpl;

template<typename Arg0 = mpl::na,
         typename Arg1 = mpl::na,
         typename Arg2 = mpl::na>
struct Vector
{
    typedef Arg0 x;
    typedef Arg1 y;
    typedef Arg2 z;

    typedef mpl::vector<x, y, z> mplVector;

    template<int element>
    struct at
    {
        typedef typename mpl::at_c<mplVector, element>::type type;
    };

    static const int dim = mpl::size<mplVector >::type::value;

    typedef typename detail::TypeSelector<x>::type type;
    typedef Vector<x, y, z> This;
    typedef math::Vector<type, dim> RT_type;
    typedef This vector_type;

    template<typename OtherType>
    HDINLINE
    operator math::Vector<OtherType, dim>() const
    {
        math::Vector<OtherType, dim> result;
        math::CT::detail::VectorFromCT<dim>()(result, *this);
        return result;
    }

    /** Create a runtime Vector
     *
     *  Creates the corresponding runtime vector
     *  object
     *
     *  \return RT_type runtime vector with same value type
     */
    static HDINLINE RT_type toRT()
    {
        math::Vector<type, dim> result;
        math::CT::detail::VectorFromCT<dim>()(result, This());
        return result;
    }
};

//*********************************************************

//________________________OperatorBase____________________________
template<typename Lhs, typename Rhs, typename T_BinaryOperator>
struct applyOperator
{
    typedef typename applyOperator<typename Lhs::vector_type,
                         typename Rhs::vector_type, T_BinaryOperator>::type type;
};

template<typename T_TypeA,
         typename T_TypeB,
         typename T_BinaryOperator>
struct applyOperator<CT::Vector<T_TypeA>, CT::Vector<T_TypeB>,T_BinaryOperator>
{
    typedef typename mpl::apply<T_BinaryOperator, T_TypeA,T_TypeB>::type OpResult;
    typedef CT::Vector<OpResult> type;
};

template<typename T_TypeA0, typename T_TypeA1,
         typename T_TypeB0, typename T_TypeB1,
         typename T_BinaryOperator>
struct applyOperator<CT::Vector<T_TypeA0, T_TypeA1>,
                     CT::Vector<T_TypeB0, T_TypeB1>,
                     T_BinaryOperator>
{
    typedef typename mpl::apply<T_BinaryOperator, T_TypeA0,T_TypeB0>::type OpResult0;
    typedef typename mpl::apply<T_BinaryOperator, T_TypeA1,T_TypeB1>::type OpResult1;
    typedef CT::Vector<OpResult0, OpResult1> type;
};

template<typename T_TypeA0, typename T_TypeA1, typename T_TypeA2,
         typename T_TypeB0, typename T_TypeB1, typename T_TypeB2,
         typename T_BinaryOperator>
struct applyOperator<CT::Vector<T_TypeA0, T_TypeA1, T_TypeA2>,
                     CT::Vector<T_TypeB0, T_TypeB1, T_TypeB2>,
                     T_BinaryOperator>
{
    typedef typename mpl::apply<T_BinaryOperator, T_TypeA0,T_TypeB0>::type OpResult0;
    typedef typename mpl::apply<T_BinaryOperator, T_TypeA1,T_TypeB1>::type OpResult1;
    typedef typename mpl::apply<T_BinaryOperator, T_TypeA2,T_TypeB2>::type OpResult2;
    typedef CT::Vector<OpResult0, OpResult1, OpResult2> type;
};

//________________________A D D____________________________

template<typename Lhs, typename Rhs>
struct add
{
    typedef typename applyOperator<
                         typename Lhs::vector_type,
                         typename Rhs::vector_type,
                         mpl::plus<mpl::_1, mpl::_2> >::type type;
};

//________________________M U L____________________________

template<typename Lhs, typename Rhs>
struct mul
{
    typedef typename applyOperator<
                         typename Lhs::vector_type,
                         typename Rhs::vector_type,
                         mpl::times<mpl::_1, mpl::_2> >::type type;
};

//________________________M A X____________________________

template<typename Lhs, typename Rhs>
struct max
{
    typedef typename applyOperator<
                         typename Lhs::vector_type,
                         typename Rhs::vector_type,
                         mpl::max<mpl::_1, mpl::_2> >::type type;
};

//________________________M I N____________________________

template<typename Lhs, typename Rhs>
struct min
{
    typedef typename applyOperator<
                         typename Lhs::vector_type,
                         typename Rhs::vector_type,
                         mpl::min<mpl::_1, mpl::_2> >::type type;
};

//________________________D O T____________________________

template<typename Lhs, typename Rhs>
struct dot
{
    typedef typename mul<Lhs,Rhs>::type MulResult;
    typedef typename mpl::accumulate<
            typename MulResult::mplVector,
            mpl::int_<0>,
            mpl::plus<mpl::_1,mpl::_2>
    >::type type;
};

//________________________V O L U M E____________________________

template<typename T_Vec>
struct volume
{
    typedef typename mpl::accumulate<
            typename T_Vec::mplVector,
            mpl::int_<1>,
            mpl::times<mpl::_1,mpl::_2>
    >::type type;
};

} // CT
} // math
} // PMacc
