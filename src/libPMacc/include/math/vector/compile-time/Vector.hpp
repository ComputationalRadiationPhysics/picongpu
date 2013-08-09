/**
 * Copyright 2013 Heiko Burau, René Widera
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
 
#ifndef STLPICCTVECTOR_HPP
#define STLPICCTVECTOR_HPP

#include <stdint.h>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/aux_/na.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/max.hpp>
#include <boost/mpl/times.hpp>
//#include <boost/mpl/arithmetic.hpp>
#include "../Vector.hpp"
#include <boost/mpl/int.hpp>

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
    template<int element>
    struct at
    {
        typedef typename mpl::at_c<mpl::vector<Arg0, Arg1, Arg2>, element>::type type;
    };
    typedef typename detail::TypeSelector<Arg0>::type type;
    typedef Vector<Arg0, Arg1, Arg2> This;
    typedef This vector_type;
    
    static const int dim = mpl::size<mpl::vector<Arg0, Arg1, Arg2> >::type::value;
    
    template<typename OtherType>
    HDINLINE
    operator math::Vector<OtherType, dim>() const
    {
        math::Vector<OtherType, dim> result;
        math::CT::detail::VectorFromCT<dim>()(result, *this);
        return result;
    }
    HDINLINE math::Vector<type, dim> vec() const
    {
        return (math::Vector<type, dim>)(*this);
    }
};

//*********************************************************

//________________________A D D____________________________
template<typename Lhs, typename Rhs, typename dummy = mpl::na>
struct add;

template<typename ArgA0,
         typename ArgB0>
struct add<CT::Vector<ArgA0>, CT::Vector<ArgB0> >
{
    typedef CT::Vector<typename mpl::plus<ArgA0, ArgB0>::type> type;
};
template<typename ArgA0, typename ArgA1,
         typename ArgB0, typename ArgB1>
struct add<CT::Vector<ArgA0, ArgA1>, CT::Vector<ArgB0, ArgB1> >
{
    typedef CT::Vector<typename mpl::plus<ArgA0, ArgB0>::type,
                       typename mpl::plus<ArgA1, ArgB1>::type> type;
};
template<typename ArgA0, typename ArgA1, typename ArgA2,
         typename ArgB0, typename ArgB1, typename ArgB2>
struct add<CT::Vector<ArgA0, ArgA1, ArgA2>, CT::Vector<ArgB0, ArgB1, ArgB2> >
{
    typedef CT::Vector<typename mpl::plus<ArgA0, ArgB0>::type,
                       typename mpl::plus<ArgA1, ArgB1>::type,
                       typename mpl::plus<ArgA2, ArgB2>::type> type;
};

template<typename Lhs, typename Rhs>
struct add<Lhs, Rhs>
{
    typedef typename add<typename Lhs::vector_type, 
                         typename Rhs::vector_type>::type type;
};

//________________________M U L____________________________

template<typename Lhs, typename Rhs, typename dummy = mpl::na>
struct mul;

template<typename ArgA0,
         typename ArgB0>
struct mul<CT::Vector<ArgA0>, CT::Vector<ArgB0> >
{
    typedef CT::Vector<typename mpl::times<ArgA0, ArgB0>::type> type;
};
template<typename ArgA0, typename ArgA1,
         typename ArgB0, typename ArgB1>
struct mul<CT::Vector<ArgA0, ArgA1>, CT::Vector<ArgB0, ArgB1> >
{
    typedef CT::Vector<typename mpl::plus<ArgA0, ArgB0>::type,
                       typename mpl::plus<ArgA1, ArgB1>::type> type;
};
template<typename ArgA0, typename ArgA1, typename ArgA2,
         typename ArgB0, typename ArgB1, typename ArgB2>
struct mul<CT::Vector<ArgA0, ArgA1, ArgA2>, CT::Vector<ArgB0, ArgB1, ArgB2> >
{
    typedef CT::Vector<typename mpl::plus<ArgA0, ArgB0>::type,
                       typename mpl::plus<ArgA1, ArgB1>::type,
                       typename mpl::plus<ArgA2, ArgB2>::type> type;
};

template<typename Lhs, typename Rhs>
struct mul<Lhs, Rhs>
{
    typedef typename mul<typename Lhs::vector_type, 
                         typename Rhs::vector_type>::type type;
};

//________________________M A X____________________________

template<typename Lhs, typename Rhs, typename dummy = mpl::na>
struct max;

template<typename ArgA0,
         typename ArgB0>
struct max<CT::Vector<ArgA0>, CT::Vector<ArgB0> >
{
    typedef CT::Vector<typename mpl::max<ArgA0, ArgB0>::type> type;
};
template<typename ArgA0, typename ArgA1,
         typename ArgB0, typename ArgB1>
struct max<CT::Vector<ArgA0, ArgA1>, CT::Vector<ArgB0, ArgB1> >
{
    typedef CT::Vector<typename mpl::plus<ArgA0, ArgB0>::type,
                       typename mpl::plus<ArgA1, ArgB1>::type> type;
};
template<typename ArgA0, typename ArgA1, typename ArgA2,
         typename ArgB0, typename ArgB1, typename ArgB2>
struct max<CT::Vector<ArgA0, ArgA1, ArgA2>, CT::Vector<ArgB0, ArgB1, ArgB2> >
{
    typedef CT::Vector<typename mpl::plus<ArgA0, ArgB0>::type,
                       typename mpl::plus<ArgA1, ArgB1>::type,
                       typename mpl::plus<ArgA2, ArgB2>::type> type;
};

template<typename Lhs, typename Rhs>
struct max<Lhs, Rhs>
{
    typedef typename max<typename Lhs::vector_type, 
                         typename Rhs::vector_type>::type type;
};

//________________________D O T____________________________

template<typename Lhs, typename Rhs, typename dummy = mpl::na>
struct dot;

template<typename ArgA0,
         typename ArgB0>
struct dot<CT::Vector<ArgA0>, CT::Vector<ArgB0> >
{
    typedef typename mul<ArgA0, ArgB0>::type type;
};
template<typename ArgA0, typename ArgA1,
         typename ArgB0, typename ArgB1>
struct dot<CT::Vector<ArgA0, ArgA1>, CT::Vector<ArgB0, ArgB1> >
{
    typedef typename add<typename mul<ArgA0, ArgB0>::type,
                         typename mul<ArgA1, ArgB1>::type>::type type;
};
template<typename ArgA0, typename ArgA1, typename ArgA2,
         typename ArgB0, typename ArgB1, typename ArgB2>
struct dot<CT::Vector<ArgA0, ArgA1, ArgA2>, CT::Vector<ArgB0, ArgB1, ArgB2> >
{
    typedef typename add<
            typename add<typename mul<ArgA0, ArgB0>::type,
                         typename mul<ArgA1, ArgB1>::type>::type,
                         typename mul<ArgA2, ArgB2>::type>::type type;
};

template<typename Lhs, typename Rhs>
struct dot<Lhs, Rhs>
{
    typedef typename dot<typename Lhs::vector_type, 
                         typename Rhs::vector_type>::type type;
};

//________________________V O L U M E____________________________

template<typename Vec, int dim = Vec::dim>
struct volume;

template<typename Vec>
struct volume<Vec, 1>
{
    typedef typename Vec::x type;
};
template<typename Vec>
struct volume<Vec, 2>
{
    typedef typename mpl::times<typename Vec::x, typename Vec::y>::type type;
};
template<typename Vec>
struct volume<Vec, 3>
{
    typedef typename mpl::times<typename Vec::x, typename Vec::y, typename Vec::z>::type type;
};

} // CT
} // math
} // PMacc

#endif //STLPICCTVECTOR_HPP
