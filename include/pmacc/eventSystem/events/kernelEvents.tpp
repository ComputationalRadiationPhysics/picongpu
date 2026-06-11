/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
            m_file,
            m_line,
            gridExtent,
            blockExtent};
    }

    template<typename T_KernelFunctor>
    template<typename T_VectorGrid, typename T_VectorBlock>
    HINLINE auto KernelPreperationWrapper<T_KernelFunctor>::operator()(
        T_VectorGrid const& gridExtent,
        T_VectorBlock const& blockExtent) const -> KernelLauncher<T_KernelFunctor, GetDim<T_VectorGrid>::dim>
    {
        return {m_kernelFunctor, m_file, m_line, gridExtent, blockExtent};
    }

} // namespace pmacc::exec::detail
