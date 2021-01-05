/* Copyright 2013-2021 Heiko Burau, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"
#include "pmacc/cuSTL/container/CartBuffer.hpp"

namespace pmacc
{
    namespace container
    {
        template<typename Type, int dim>
        struct PseudoBuffer : public container::CartBuffer<Type, dim>
        {
            template<typename _Type>
            PseudoBuffer(pmacc::DeviceBuffer<_Type, dim>& devBuffer);
            template<typename _Type>
            PseudoBuffer(pmacc::HostBuffer<_Type, dim>& hostBuffer);
        };

    } // namespace container
} // namespace pmacc

#include "PseudoBuffer.tpp"
