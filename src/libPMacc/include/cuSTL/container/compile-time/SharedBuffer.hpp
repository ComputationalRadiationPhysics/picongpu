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
 
#ifndef CONTAINER_CT_SHAREDBUFFER_HPP
#define CONTAINER_CT_SHAREDBUFFER_HPP

#include "CartBuffer.hpp"
#include "../allocator/compile-time/SharedMemAllocator.hpp"

namespace PMacc
{
namespace container
{
namespace CT
{
    
template<typename Type, typename Size, int uid = 0>
struct SharedBuffer 
 : public CT::CartBuffer<Type, Size, 
                         allocator::CT::SharedMemAllocator<Type, Size, Size::dim, uid>, void, void>
{};
    
} // CT
} // container
} // PMacc

#endif // CONTAINER_CT_SHAREDBUFFER_HPP
