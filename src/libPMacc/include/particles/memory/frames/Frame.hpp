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
#include "particles/factories/CreateMap.hpp"
#include <boost/utility/result_of.hpp>
#include <boost/mpl/find.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/deref.hpp>

#include "particles/factories/GetKeyFromAlias.hpp"

#include "traits/HasIdentifier.hpp"


namespace PMacc
{
namespace bmpl = boost::mpl;
namespace pmath = PMacc::math;
namespace pmacc = PMacc;

template<template<typename> class T_CreatePairOperator, 
        typename T_ValueTypeSeq, 
        typename T_MethodsList = bmpl::list<>,
        typename T_Flags = bmpl::list<> >
struct Frame
:
protected pmath::MapTuple<typename CreateMap<T_ValueTypeSeq, T_CreatePairOperator>::type, pmath::AlignedData>
{
    // typedef T_CreatePairOperator CoverOperator;
    typedef T_ValueTypeSeq ValueTypeSeq;
    typedef T_MethodsList MethodsList;
    typedef T_Flags AttributeList;
    typedef Frame<T_CreatePairOperator, ValueTypeSeq, MethodsList, AttributeList> ThisType;
    typedef pmath::MapTuple<typename CreateMap<ValueTypeSeq, T_CreatePairOperator>::type, pmath::AlignedData> BaseType;

    typedef pmacc::Particle<ThisType, MethodsList> ParticleType;

    HDINLINE ParticleType operator[](const uint32_t idx)
    {
        return ParticleType(*this, idx);
    }


    HDINLINE ParticleType operator[](const uint32_t idx) const
    {
        return ParticleType(*this, idx);
    }

    template<typename T_Key >
        HDINLINE
        typename boost::result_of < BaseType(typename GetKeyFromAlias_assert<ValueTypeSeq,T_Key>::type)>::type
    getIdentifier(const T_Key) const
    {      
        typedef typename GetKeyFromAlias<ValueTypeSeq,T_Key>::type Key;
        return BaseType::operator[](Key());
    }

    template<typename T_Key >
        HDINLINE
        typename boost::result_of < BaseType(typename GetKeyFromAlias_assert<ValueTypeSeq,T_Key>::type)>::type
    getIdentifier(const T_Key)
    {
        typedef typename GetKeyFromAlias<ValueTypeSeq,T_Key>::type Key;
        return BaseType::operator[](Key());
    }
};

namespace traits
{

template<typename T_Key,
template<typename> class T_CreatePairOperator,
typename T_ValueTypeSeq,
typename T_MethodsList,
typename T_Flags
>
struct HasIdentifier<
PMacc::Frame<T_CreatePairOperator, T_ValueTypeSeq, T_MethodsList, T_Flags>,
T_Key
>
{
private:
    typedef PMacc::Frame<T_CreatePairOperator, T_ValueTypeSeq, T_MethodsList, T_Flags> FrameType;
public:
    typedef typename FrameType::ValueTypeSeq ValueTypeSeq;
    typedef typename GetKeyFromAlias<ValueTypeSeq,T_Key>::type Key;
    /* if Key is void_ than we have no T_Key in our Sequence.
     * checks also to alias keys
     */
    typedef bmpl::bool_<!boost::is_same< bmpl::void_,Key>::value> type;
    static const bool value = type::value;
};
} //namespace traits

}//namespace PMacc
