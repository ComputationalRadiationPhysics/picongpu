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

#include <boost/mpl/if.hpp>

#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>


namespace PMacc
{
namespace algorithms
{
namespace forEachFunctor
{
namespace detail
{

struct DoNothing
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

template< typename itA, typename itEnd, typename Accessor>
struct ForEach;

template< typename itA, typename itEnd, template<typename> class Accessor_, typename A1>
struct ForEach< itA, itEnd, Accessor_<A1> >
{
    typedef typename boost::mpl::next<itA>::type next;
    typedef typename boost::mpl::deref<itA>::type usedType;
    typedef typename boost::is_same<next, itEnd>::type isEnd;
    typedef Accessor_<A1> Accessor;
    typedef Accessor_<usedType> AccessorType;

    typedef detail::ForEach< next, itEnd, Accessor > tmpNextCall;
    typedef typename boost::mpl::if_< isEnd, typename detail::DoNothing, tmpNextCall>::type nextCall;

    HDINLINE void operator()() const
    {
        // execute the Accessor
        AccessorType()();
        // go until itEnd
        nextCall()();
    }

    template<typename T1 >
        HDINLINE void operator()(T1 &t1) const
    {
        // execute the Accessor
        AccessorType()(t1);
        // go until itEnd
        nextCall()(t1);
    }

    template<typename T1, typename T2 >
        HDINLINE void operator()(T1 &t1, T2 &t2) const
    {
        // execute the Accessor
        AccessorType()(t1, t2);
        // go until itEnd
        nextCall()(t1, t2);
    }
};

template< typename itA, typename itEnd, template<typename, typename> class Accessor_, typename A1, typename A2>
struct ForEach<itA, itEnd, Accessor_<A1, A2> >
{
    typedef typename boost::mpl::next<itA>::type next;
    typedef typename boost::mpl::deref<itA>::type usedType;
    typedef typename boost::is_same<next, itEnd>::type isEnd;
    typedef Accessor_<A1, A2> Accessor;
    typedef Accessor_<usedType, A2> AccessorType;

    typedef detail::ForEach< next, itEnd, Accessor > tmpNextCall;
    typedef typename boost::mpl::if_<isEnd, detail::DoNothing, tmpNextCall>::type nextCall;

    HDINLINE void operator()() const
    {
        // execute the Accessor
        AccessorType()();
        // go until itEnd
        nextCall()();
    }

    template<typename T1 >
        HDINLINE void operator()(T1 &t1) const
    {
        // execute the Accessor
        AccessorType()(t1);
        // go until itEnd
        nextCall()(t1);
    }

    template<typename T1, typename T2 >
        HDINLINE void operator()(T1 &t1, T2 &t2) const
    {
        // execute the Accessor
        AccessorType()(t1, t2);
        // go until itEnd
        nextCall()(t1, t2);
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

    typedef typename boost::is_same<begin, end>::type isEnd;
    typedef detail::ForEach< begin, end, Accessor > tmpNextCall;
    typedef typename boost::mpl::if_<isEnd, detail::DoNothing, tmpNextCall>::type nextCall;

    HDINLINE void operator()() const
    {
        nextCall()();
    }

    template<typename T1 >
        HDINLINE void operator()(T1 &t1) const
    {
        nextCall()(t1);
    }

    template<typename T1, typename T2 >
        HDINLINE void operator()(T1 &t1, T2 &t2) const
    {
        nextCall(t1, t2);
    }
};

} // namespace forEachFunctor
} // namespace algorithms
} // namespace PMacc