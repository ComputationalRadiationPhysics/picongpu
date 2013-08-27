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

#include <boost/mpl/map.hpp>
#include <boost/mpl/list.hpp>
#include "math/MapTuple.hpp"


#include <boost/type_traits.hpp>

#include "particles/memory/dataTypes/Particle.hpp"
#include "particles/frame_types.hpp"
#include "particles/factories/CoverTypes.hpp"
#include <boost/utility/result_of.hpp>

namespace PMacc
{
namespace bmpl = boost::mpl;
namespace pmath = PMacc::math;
namespace pmacc = PMacc;

template<template<typename> class CoverOperator_, typename ValueTypeMap_, typename MethodsList_ = bmpl::list<>, typename AttributeList_ = bmpl::list<> >
struct Frame
:
public pmath::MapTuple<typename CoverTypes<ValueTypeMap_, CoverOperator_>::type,pmath::AlignedData>
{
    // typedef CoverOperator_ CoverOperator;
    typedef ValueTypeMap_ ValueTypeMap;
    typedef MethodsList_ MethodsList;
    typedef AttributeList_ AttributeList;
    typedef Frame<CoverOperator_, ValueTypeMap, MethodsList, AttributeList> ThisType;
    typedef pmath::MapTuple<typename CoverTypes<ValueTypeMap_, CoverOperator_>::type,pmath::AlignedData> BaseType;

    typedef pmacc::Particle<ThisType, MethodsList> ParticleType;


    HDINLINE ParticleType operator[](const uint32_t idx)
    {
        return ParticleType(*this, idx);
    }

    template<typename TKey >
        HDINLINE
        typename boost::result_of < BaseType(TKey)>::type
        operator()(const TKey) const
    {
        return BaseType::operator[](TKey());
    }

    template<typename TKey >
        HDINLINE
        typename boost::result_of < BaseType(TKey)>::type
        operator()(const TKey)
    {
        return BaseType::operator[](TKey());
    }
};

}//namespace PMacc
