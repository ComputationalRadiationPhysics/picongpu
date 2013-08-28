/**
 * Copyright 2013 Axel Huebl, René Widera
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

#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/next_prior.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/bind.hpp>
 #include <boost/type_traits.hpp>


namespace PMacc
{
namespace algorithms
{
namespace forEachFunctor
{
namespace detail
{
template<bool isEnd, typename itA, typename itEnd, typename Accessor>
struct ForEach;

template< typename itA, typename itEnd, template<typename> class Accessor_,typename A1>
struct ForEach<false,itA,itEnd,Accessor_<A1> >
{
    typedef typename boost::mpl::next<itA>::type next;
    typedef typename boost::mpl::deref<itA>::type usedType;
    static const bool isEnd=boost::is_same<next,itEnd>::value;
    typedef Accessor_<A1> Accessor;
    typedef Accessor_<usedType> AccessorType;

    HDINLINE void operator()() const
    {
        // execute the Accessor
        AccessorType()();
        // go until itEnd
        detail::ForEach<isEnd,next, itEnd, Accessor >()();
    }

    template<typename T1 >
        HDINLINE void operator()(T1 &t1) const
    {
        // execute the Accessor
        AccessorType()(t1);
        // go until itEnd
        detail::ForEach<isEnd,next, itEnd, Accessor >()(t1);
    }

    template<typename T1, typename T2 >
        HDINLINE void operator()(T1 &t1, T2 &t2) const
    {
        // execute the Accessor
        AccessorType()(t1, t2);
        // go until itEnd
        detail::ForEach<isEnd,next, itEnd, Accessor >()(t1, t2);
    }
};

template< typename itA, typename itEnd, template<typename,typename> class Accessor_,typename A1,typename A2>
struct ForEach<false,itA,itEnd,Accessor_<A1,A2> >
{
    typedef typename boost::mpl::next<itA>::type next;
    typedef typename boost::mpl::deref<itA>::type usedType;
    static const bool isEnd=boost::is_same<next,itEnd>::value;
    typedef Accessor_<A1,A2> Accessor;
    typedef Accessor_<usedType,A2> AccessorType;

    HDINLINE void operator()() const
    {
        // execute the Accessor
        AccessorType()();
        // go until itEnd
        detail::ForEach<isEnd,next, itEnd, Accessor >()();
    }

    template<typename T1 >
        HDINLINE void operator()(T1 &t1) const
    {
        // execute the Accessor
        AccessorType()(t1);
        // go until itEnd
        detail::ForEach<isEnd,next, itEnd, Accessor >()(t1);
    }

    template<typename T1, typename T2 >
        HDINLINE void operator()(T1 &t1, T2 &t2) const
    {
        // execute the Accessor
        AccessorType()(t1, t2);
        // go until itEnd
        detail::ForEach<isEnd,next, itEnd, Accessor >()(t1, t2);
    }
};

template< typename itEnd, typename Accessor>
struct ForEach<true,itEnd, itEnd, Accessor>
{

    HDINLINE void operator()() const
    {
    }

    template<typename T1 >
        HDINLINE void operator()(T1) const
    {
    }

    template<typename T1, typename T2 >
        HDINLINE void operator()(T1, T2) const
    {
    }
};

} // namespace detail



/** Compile-Time for each for Boost::MPL Type Lists
 * 
 *  \tparam MPLTypeList A TypeList which can be accessed by begin, end, next
 *  \tparam Accessor A type that implements evaluate<T>() as a
 *          static void member for HDINLINE
 */
template< typename MPLTypeList, typename Accessor >
struct ForEach
{
    typedef typename boost::mpl::begin<MPLTypeList>::type begin;
    typedef typename boost::mpl::end< MPLTypeList>::type end;
    static const bool isEnd=boost::is_same<begin,end>::value;

    HDINLINE void operator()() const
    {
        return detail::ForEach<isEnd,begin, end, Accessor >()();
    }

    template<typename T1 >
        HDINLINE void operator()(T1 &t1) const
    {
        detail::ForEach<isEnd,begin, end, Accessor >()(t1);
    }

    template<typename T1 ,typename T2>
        HDINLINE void operator()( T1 &t1, T2 &t2) const
    {
        detail::ForEach<isEnd,begin, end, Accessor >()(t1, t2);
    }
};

} // namespace forEachFunctor
} // namespace algorithms
} // namespace PMacc