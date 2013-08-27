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

namespace PMacc
{
namespace bmpl = boost::mpl;
namespace pmath = PMacc::math;
namespace pmacc = PMacc;


template<typename FrameType_, typename MethodsList_>
struct Particle : public bmpl::inherit<MethodsList_>::type
{
    typedef FrameType_ FrameType;
    typedef MethodsList_ MethodsList;

    FrameType& frame;
    uint32_t idx;

    HDINLINE Particle(FrameType& frame, uint32_t idx) : frame(frame), idx(idx)
    {
    }

    template<typename TKey >
        HDINLINE
        typename boost::result_of<
        typename boost::remove_reference<
        typename boost::result_of < FrameType(TKey)>::type
        >::type(uint32_t)
    >::type
        operator[](const TKey key)
    {
        return frame[key][idx];
    }

    template<typename TKey >
        HDINLINE
        typename boost::result_of<
        typename boost::remove_reference<
        typename boost::result_of < FrameType(TKey)>::type
        >::type(uint32_t)
    >::type
        operator[](const TKey key) const
    {
        return frame[key][idx];
    }

};

} //namespace PMacc
