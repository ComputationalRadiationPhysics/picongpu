/* Copyright 2016-2021 Rene Widera
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


#include "pmacc/types.hpp"
#include "pmacc/eventSystem/events/kernelEvents.hpp"


namespace pmacc
{
    namespace exec
    {
        template<typename T_KernelFunctor>
        template<typename T_VectorGrid, typename T_VectorBlock>
        HINLINE auto Kernel<T_KernelFunctor>::operator()(
            T_VectorGrid const& gridExtent,
            T_VectorBlock const& blockExtent,
            size_t const sharedMemByte) const -> KernelStarter<Kernel, T_VectorGrid, T_VectorBlock>
        {
            return KernelStarter<Kernel, T_VectorGrid, T_VectorBlock>(*this, gridExtent, blockExtent, sharedMemByte);
        }
    } // namespace exec
} // namespace pmacc
