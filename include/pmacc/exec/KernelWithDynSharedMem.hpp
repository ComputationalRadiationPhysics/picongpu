/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once


#include <string>

namespace pmacc::exec::detail
{
    /** Kernel with dynamic shared memory
     *
     * This implements the possibility to define dynamic shared memory without
     * specializing the needed alpaka trait BlockSharedMemDynSizeBytes for each kernel with shared memory.
     * The trait BlockSharedMemDynSizeBytes is defined by PMacc for all types of KernelWithDynSharedMem.
     */
    template<typename T_Kernel>
    struct KernelWithDynSharedMem : public T_Kernel
    {
        size_t const m_dynSharedMemBytes;

        KernelWithDynSharedMem(T_Kernel const& kernel, size_t const& dynSharedMemBytes)
            : T_Kernel(kernel)
            , m_dynSharedMemBytes(dynSharedMemBytes)
        {
        }
    };
} // namespace pmacc::exec::detail
