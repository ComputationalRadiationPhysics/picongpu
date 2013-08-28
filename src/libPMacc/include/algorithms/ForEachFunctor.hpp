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


namespace PMacc
{
namespace algorithms
{
namespace forEachFunctor
{

template<typename T>
struct getOptimalType;

template<typename T>
struct getOptimalType<const T>
{
    typedef const T type;
};

template<typename T>
struct getOptimalType<const T&>
{
    typedef const T& type;
};

template<typename T>
struct getOptimalType< T&>
{
    typedef T& type;
};

namespace detail
{

template< typename itA, typename itEnd, template<typename> class Accessor>
struct ForEach
{
    typedef typename boost::mpl::next<itA>::type next;
    typedef typename boost::mpl::deref<itA>::type usedType;
    

    HDINLINE void operator()() const
    {
        // execute the Accessor
        Accessor<usedType >()();
        // go until itEnd
        detail::ForEach<next, itEnd, Accessor >()();
    }

    template<typename T1 >
        HDINLINE void operator()(T1 &t1) const
    {
        // execute the Accessor
        Accessor<usedType >()(t1);
        // go until itEnd
        detail::ForEach<next, itEnd, Accessor >()(t1);
    }

    template<typename T1, typename T2 >
        HDINLINE void operator()(T1 &t1, T2 &t2) const
    {
        // execute the Accessor
        Accessor<usedType >()(t1, t2);
        // go until itEnd
        detail::ForEach<next, itEnd, Accessor >()(t1, t2);
    }
};

template< typename itEnd, template<typename> class Accessor>
struct ForEach<itEnd, itEnd, Accessor>
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
template< typename MPLTypeList, template<typename> class Accessor >
struct ForEach
{
    typedef typename boost::mpl::begin<MPLTypeList>::type begin;
    typedef typename boost::mpl::end< MPLTypeList>::type end;

    HDINLINE void operator()() const
    {
        return detail::ForEach<begin, end, Accessor >()();
    }

    template<typename T1 >
        HDINLINE void operator()(T1 &t1) const
    {
        detail::ForEach<begin, end, Accessor >()(t1);
    }

    template<typename T1 ,typename T2>
        HDINLINE void operator()( T1 &t1, T2 &t2) const
    {
        detail::ForEach<begin, end, Accessor >()(t1, t2);
    }
};

} // namespace forEachFunctor
} // namespace algorithms
} // namespace PMacc