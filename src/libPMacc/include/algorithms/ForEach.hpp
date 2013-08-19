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

#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/next_prior.hpp>
#include <boost/mpl/deref.hpp>


namespace PMacc
{
namespace algorithms
{
namespace forEach
{
    namespace detail
    {
        template< typename itA, typename itEnd, typename Accessor>
        struct ForEach
        {
            typedef typename boost::mpl::next<itA>::type next;
            
            HDINLINE static void evaluate()
            {
                // execute the Accessor
                Accessor::template evaluate< typename boost::mpl::deref<itA>::type >();
                
                // go until itEnd
                detail::ForEach<next, itEnd, Accessor>::evaluate();
            }
        };
        
        template< typename itEnd, typename Accessor>
        struct ForEach<itEnd, itEnd, Accessor>
        {
            HDINLINE static void evaluate()
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
        HDINLINE void operator()( ) const
        {
            typedef typename boost::mpl::begin<MPLTypeList>::type begin;
            typedef typename boost::mpl::end<  MPLTypeList>::type end;
        
            return detail::ForEach<begin, end, Accessor>::evaluate();
        }
    };
    
} // namespace forEach
} // namespace algorithms
} // namespace PMacc