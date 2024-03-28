/* Copyright 2016-2023 Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


#include "pmacc/eventSystem/events/kernelEvents.hpp"
#include "pmacc/types.hpp"

#include <cstdint>

namespace pmacc::exec::detail
{
    template<typename T_KernelFunctor>
    template<typename T_VectorGrid, typename T_VectorBlock>
    HINLINE auto KernelPreperationWrapper<T_KernelFunctor>::operator()(
        T_VectorGrid const& gridExtent,
        T_VectorBlock const& blockExtent,
        size_t const sharedMemByte) const
        -> KernelLauncher<KernelWithDynSharedMem<T_KernelFunctor>, GetDim<T_VectorGrid>::dim>
    {
        return {
            KernelWithDynSharedMem<T_KernelFunctor>(m_kernelFunctor, sharedMemByte),
            m_metaData,
            gridExtent,
            blockExtent};
    }

    template<typename T_KernelFunctor>
    template<typename T_VectorGrid, typename T_VectorBlock>
    HINLINE auto KernelPreperationWrapper<T_KernelFunctor>::operator()(
        T_VectorGrid const& gridExtent,
        T_VectorBlock const& blockExtent) const -> KernelLauncher<T_KernelFunctor, GetDim<T_VectorGrid>::dim>
    {
        return {m_kernelFunctor, m_metaData, gridExtent, blockExtent};
    }

} // namespace pmacc::exec::detail
