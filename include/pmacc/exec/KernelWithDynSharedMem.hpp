/* Copyright 2022-2023 Rene Widera
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
