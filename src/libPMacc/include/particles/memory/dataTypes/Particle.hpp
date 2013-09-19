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
#include <boost/utility/result_of.hpp>
#include <boost/mpl/inherit.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>
#include "traits/HasIdentifier.hpp"

namespace PMacc
{
namespace bmpl = boost::mpl;
namespace pmath = PMacc::math;
namespace pmacc = PMacc;

template<typename T_FrameType, typename T_MethodsList>
struct Particle : public bmpl::inherit<T_MethodsList>::type
{
    typedef T_FrameType FrameType;
    typedef T_MethodsList MethodsList;
    typedef typename FrameType::ValueTypeSeq ValueTypeSeq;

    FrameType& frame;
    uint32_t idx;

    HDINLINE Particle(FrameType& frame, uint32_t idx) : frame(frame), idx(idx)
    {
    }

    template<typename T_Key >
        HDINLINE
        typename boost::result_of<
        typename boost::remove_reference<
        typename boost::result_of < FrameType(T_Key)>::type
        >::type(uint32_t)
    >::type
        operator[](const T_Key key)
        {
            return frame.getIdentifier(key)[idx];
        }

    template<typename T_Key >
        HDINLINE
        typename boost::result_of<
        typename boost::remove_reference<
        typename boost::result_of < FrameType(T_Key)>::type
        >::type(uint32_t)
    >::type
        operator[](const T_Key key) const
        {
            return frame.getIdentifier(key)[idx];
        }

};

namespace traits
{

template<typename T_Key,
typename T_FrameType,
typename T_MethodsList
>
struct HasIdentifier<
PMacc::Particle<T_FrameType, T_MethodsList>,
T_Key
>
{
private:
    typedef T_FrameType FrameType;
public:
    typedef typename HasIdentifier<FrameType, T_Key>::type type;
    static const bool value = type::value;
};
} //namespace traits

} //namespace PMacc
