/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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

#ifndef CONTAINER_PSEUDOBUFFER_HPP
#define CONTAINER_PSEUDOBUFFER_HPP

#include "memory/buffers/DeviceBuffer.hpp"
#include "memory/buffers/HostBuffer.hpp"
#include "cuSTL/container/CartBuffer.hpp"

namespace PMacc
{
namespace container
{

template<typename Type, int dim>
struct PseudoBuffer : public container::CartBuffer<Type, dim>
{
    template<typename _Type>
    PseudoBuffer(PMacc::DeviceBuffer<_Type, dim>& devBuffer);
    template<typename _Type>
    PseudoBuffer(PMacc::HostBuffer<_Type, dim>& hostBuffer);
};

} // container
} // PMacc

#include "PseudoBuffer.tpp"

#endif // CONTAINER_PSEUDOBUFFER_HPP
