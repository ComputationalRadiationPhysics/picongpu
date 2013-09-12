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
 
#pragma once

#include <types.h>
#include <math/vector/Int.hpp>
#include <math/vector/compile-time/Int.hpp>
#include <forward.hpp>

namespace PMacc
{
namespace algorithm
{
namespace cudaBlock
{
    
#ifndef FOREACH_KERNEL_MAX_PARAMS
#define FOREACH_KERNEL_MAX_PARAMS 4
#endif
    
#define SHIFTACCESS_CURSOR(Z, N, _) forward(c ## N [pos])
    
#define FOREACH_OPERATOR(Z, N, _) \
    template<typename Zone, BOOST_PP_ENUM_PARAMS(N, typename C), typename Functor> \
    DINLINE void operator()(Zone, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), const Functor& functor) \
    { \
        BOOST_AUTO(functor_, lambda::make_Functor(functor)); \
        const int dataVolume = math::CT::volume<typename Zone::Size>::type::value; \
        const int blockVolume = math::CT::volume<BlockDim>::type::value; \
        for(int i = this->linearThreadIdx; i < dataVolume; i += blockVolume) \
        { \
            math::Int<3> pos( \
                Zone::Offset::x::value + (i % Zone::Size::x::value), \
                Zone::Offset::y::value + ((i % (Zone::Size::x::value * Zone::Size::y::value)) / Zone::Size::x::value), \
                Zone::Offset::z::value + (i / (Zone::Size::x::value * Zone::Size::y::value))); \
            functor_(BOOST_PP_ENUM(N, SHIFTACCESS_CURSOR, _)); \
        } \
    }
    
template<typename BlockDim>
struct Foreach
{
private:
    const int linearThreadIdx;
public:
    DINLINE Foreach() 
     : linearThreadIdx(
        threadIdx.z * BlockDim::x::value * BlockDim::y::value +
        threadIdx.y * BlockDim::x::value +
        threadIdx.x) {}
    DINLINE Foreach(int linearThreadIdx) : linearThreadIdx(linearThreadIdx) {}
    
    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_KERNEL_MAX_PARAMS), FOREACH_OPERATOR, _)
};

#undef SHIFTACCESS_CURSOR
#undef FOREACH_OPERATOR

} // cudaBlock
} // algorithm
} // PMacc
