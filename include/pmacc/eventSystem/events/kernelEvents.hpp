/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Benjamin Worpitz
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
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/traits/GetNComponents.hpp"
#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/Environment.hpp"
#include "pmacc/nvidia/gpuEntryFunction.hpp"

#include <string>


/* No namespace in this file since we only declare macro defines */

/*if this flag is defined all kernel calls would be checked and synchronize
 * this flag must set by the compiler or inside of the Makefile
 */
#if(PMACC_SYNC_KERNEL == 1)
#    define CUDA_CHECK_KERNEL_MSG(...) CUDA_CHECK_MSG(__VA_ARGS__)
#else
/*no synchronize and check of kernel calls*/
#    define CUDA_CHECK_KERNEL_MSG(...) ;
#endif


namespace pmacc
{
    namespace exec
    {
        /** configured kernel object
         *
         * this objects contains the functor and the starting parameter
         *
         * @tparam T_Kernel pmacc Kernel object
         * @tparam T_VectorGrid type which defines the grid extents (type must be castable to cupla dim3)
         * @tparam T_VectorBlock type which defines the block extents (type must be castable to cupla dim3)
         */
        template<typename T_Kernel, typename T_VectorGrid, typename T_VectorBlock>
        struct KernelStarter;

        /** wrapper for the user kernel functor
         *
         * contains debug information like filename and line of the kernel call
         */
        template<typename T_KernelFunctor>
        struct Kernel
        {
            using KernelType = T_KernelFunctor;
            /** functor */
            T_KernelFunctor const m_kernelFunctor;
            /** file name from where the kernel is called */
            std::string const m_file;
            /** line number in the file */
            size_t const m_line;

            /**
             *
             * @param gridExtent grid extent configuration for the kernel
             * @param blockExtent block extent configuration for the kernel
             * @param sharedMemByte dynamic shared memory used by the kernel (in byte )
             * @return
             */
            HINLINE Kernel(
                T_KernelFunctor const& kernelFunctor,
                std::string const& file = std::string(),
                size_t const line = 0)
                : m_kernelFunctor(kernelFunctor)
                , m_file(file)
                , m_line(line)
            {
            }

            /** configured kernel object
             *
             * this objects contains the functor and the starting parameter
             *
             * @tparam T_VectorGrid type which defines the grid extents (type must be castable to cupla dim3)
             * @tparam T_VectorBlock type which defines the block extents (type must be castable to cupla dim3)
             *
             * @param gridExtent grid extent configuration for the kernel
             * @param blockExtent block extent configuration for the kernel
             * @param sharedMemByte dynamic shared memory used by the kernel (in byte)
             */
            template<typename T_VectorGrid, typename T_VectorBlock>
            HINLINE auto operator()(
                T_VectorGrid const& gridExtent,
                T_VectorBlock const& blockExtent,
                size_t const sharedMemByte = 0) const -> KernelStarter<Kernel, T_VectorGrid, T_VectorBlock>;
        };


        template<typename T_Kernel, typename T_VectorGrid, typename T_VectorBlock>
        struct KernelStarter
        {
            /** kernel functor */
            T_Kernel const m_kernel;
            /** grid extents for the kernel */
            T_VectorGrid const m_gridExtent;
            /** block extents for the kernel */
            T_VectorBlock const m_blockExtent;
            /** dynamic shared memory consumed by the kernel (in byte) */
            size_t const m_sharedMemByte;

            /** kernel starter object
             *
             * @param kernel pmacc Kernel
             */
            HINLINE KernelStarter(
                T_Kernel const& kernel,
                T_VectorGrid const& gridExtent,
                T_VectorBlock const& blockExtent,
                size_t const sharedMemByte)
                : m_kernel(kernel)
                , m_gridExtent(gridExtent)
                , m_blockExtent(blockExtent)
                , m_sharedMemByte(sharedMemByte)
            {
            }

            /** execute the kernel functor
             *
             * @tparam T_Args types of the arguments
             * @param args arguments for the kernel functor
             *
             * @{
             */
            template<typename... T_Args>
            HINLINE void operator()(T_Args const&... args) const
            {
                std::string const kernelName = typeid(m_kernel.m_kernelFunctor).name();
                std::string const kernelInfo = kernelName + std::string(" [") + m_kernel.m_file + std::string(":")
                    + std::to_string(m_kernel.m_line) + std::string(" ]");

                CUDA_CHECK_KERNEL_MSG(cuplaDeviceSynchronize(), std::string("Crash before kernel call ") + kernelInfo);

                pmacc::TaskKernel* taskKernel
                    = pmacc::Environment<>::get().Factory().createTaskKernel(typeid(kernelName).name());

                DataSpace<traits::GetNComponents<T_VectorGrid>::value> gridExtent(m_gridExtent);

                DataSpace<traits::GetNComponents<T_VectorBlock>::value> blockExtent(m_blockExtent);

                CUPLA_KERNEL(typename T_Kernel::KernelType)
                (gridExtent.toDim3(), blockExtent.toDim3(), m_sharedMemByte, taskKernel->getCudaStream())(args...);
                CUDA_CHECK_KERNEL_MSG(
                    cuplaGetLastError(),
                    std::string("Last error after kernel launch ") + kernelInfo);
                CUDA_CHECK_KERNEL_MSG(
                    cuplaDeviceSynchronize(),
                    std::string("Crash after kernel launch ") + kernelInfo);
                taskKernel->activateChecks();
                CUDA_CHECK_KERNEL_MSG(
                    cuplaDeviceSynchronize(),
                    std::string("Crash after kernel activation") + kernelInfo);
            }

            template<typename... T_Args>
            HINLINE void operator()(T_Args const&... args)
            {
                return static_cast<const KernelStarter&>(*this)(args...);
            }

            /** @} */
        };


        /** creates a kernel object
         *
         * @tparam T_KernelFunctor type of the kernel functor
         * @param kernelFunctor instance of the functor
         * @param file file name (for debug)
         * @param line line number in the file (for debug)
         */
        template<typename T_KernelFunctor>
        auto kernel(
            T_KernelFunctor const& kernelFunctor,
            std::string const& file = std::string(),
            size_t const line = 0) -> Kernel<T_KernelFunctor>
        {
            return Kernel<T_KernelFunctor>(kernelFunctor, file, line);
        }
    } // namespace exec
} // namespace pmacc


/** create a kernel object out of a functor instance
 *
 * this macro add the current filename and line number to the kernel object
 *
 * @param ... instance of kernel functor
 */
#define PMACC_KERNEL(...) ::pmacc::exec::kernel(__VA_ARGS__, __FILE__, static_cast<size_t>(__LINE__))


#include "pmacc/eventSystem/events/kernelEvents.tpp"
