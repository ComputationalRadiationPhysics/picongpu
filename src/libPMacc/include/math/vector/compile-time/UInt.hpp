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
 
#ifndef STLPICCTUINT_HPP
#define STLPICCTUINT_HPP

#include <stdint.h>
#include "Vector.hpp"
#include <boost/mpl/integral_c.hpp>

namespace PMacc
{
namespace math
{
namespace CT
{
    
template<uint32_t x = (uint32_t)-1, uint32_t y = (uint32_t)-1, uint32_t z = (uint32_t)-1,
         uint32_t dummy = (uint32_t)-1>
struct UInt;

template<>
struct UInt<> : public CT::Vector<>
{};

template<uint32_t x>
struct UInt<x> : public CT::Vector< mpl::integral_c<uint32_t, x> >
{};

template<uint32_t x, uint32_t y>
struct UInt<x, y> : public CT::Vector<mpl::integral_c<uint32_t, x>,
                                                mpl::integral_c<uint32_t, y> >
{};

template<uint32_t x, uint32_t y, uint32_t z>
struct UInt<x, y, z> : public CT::Vector<mpl::integral_c<uint32_t, x>,
                                                   mpl::integral_c<uint32_t, y>,
                                                   mpl::integral_c<uint32_t, z> >
{};
    
} // CT
} // math
} // PMacc

#endif //STLPICCTUINT_HPP
