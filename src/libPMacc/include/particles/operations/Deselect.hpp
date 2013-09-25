/**
 * Copyright 2013 René Widera
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

#include "types.h"
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/vector.hpp>

namespace PMacc
{
namespace particles
{
namespace operations
{

namespace detail
{

template<typename T_Sequence, typename T_Object>
struct Deselect;

}//namespace detail

namespace wrapperCall
{
namespace bmpl = boost::mpl;

template<typename T_IsSequence, typename T_Sequence, typename T_Object>
struct DeselectWrapper;

template<typename T_Sequence, typename T_Object>
struct DeselectWrapper<boost::mpl::true_, T_Sequence, T_Object>
{
    BOOST_MPL_ASSERT((boost::mpl::is_sequence< T_Sequence >));
    typedef  PMacc::particles::operations::detail::Deselect<T_Sequence, T_Object> BaseType;
    typedef typename BaseType::result result;

    HDINLINE
    result operator()(const T_Object& object)
    {
        return BaseType()(object);
    }
};

template<typename T_Key, typename T_Object>
struct DeselectWrapper<boost::mpl::false_, T_Key, T_Object>
{
    typedef PMacc::particles::operations::detail::Deselect<bmpl::vector<T_Key>, T_Object> BaseType;
    typedef typename BaseType::result result;

    HDINLINE
    result operator()(const T_Object& object)
    {
        return BaseType()(object);
    }
};
} //namespace wrapperCall

template<typename T_Exclude, typename T_Object>
static HDINLINE
typename wrapperCall::DeselectWrapper<typename boost::mpl::is_sequence< T_Exclude >::type, T_Exclude, T_Object>::result
deselect(const T_Object& object)
{
    typedef typename boost::mpl::is_sequence< T_Exclude >::type IsSequence;
    typedef wrapperCall::DeselectWrapper<IsSequence, T_Exclude, T_Object> BaseType;
    typedef typename BaseType::result result;
    return BaseType()(object);
}

}//operators
}//namespace particles
} //namespace PMacc
