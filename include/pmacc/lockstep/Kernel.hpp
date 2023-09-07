/* Copyright 2017-2023 Rene Widera
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


#include "pmacc/eventSystem/events/kernelEvents.hpp"
#include "pmacc/exec/KernelLauncher.hpp"
#include "pmacc/exec/KernelMetaData.hpp"
#include "pmacc/exec/KernelWithDynSharedMem.hpp"
#include "pmacc/lockstep/Worker.hpp"
#include "pmacc/lockstep/WorkerCfg.hpp"
#include "pmacc/types.hpp"

namespace pmacc::lockstep
{
    namespace exec
    {
        namespace detail
        {
            template<typename T_Kernel, typename T_WorkerCfg>
            struct LockStepKernel : private T_Kernel
            {
                LockStepKernel(T_Kernel const& kernel) : T_Kernel(kernel)
                {
                }

                /** Forward arguments to the user kernel.
                 *
                 * The alpaka accelerator is replaced by
                 *
                 * @tparam T_Acc alpaka accelerator type
                 * @tparam T_Args user kernel argument types
                 * @param acc alpaka accelerator
                 * @param args user kernel arguments
                 */
                template<typename T_Acc, typename... T_Args>
                HDINLINE void operator()(T_Acc const& acc, T_Args&&... args) const
                {
                    auto const worker = lockstep::detail::WorkerCfgAssume1DBlock::getWorker<T_WorkerCfg>(acc);
                    T_Kernel::template operator()(worker, std::forward<T_Args>(args)...);
                }
            };


            /** Wraps a user kernel functor to prepare the execution on the device.
             *
             * This objects contains the kernel functor, kernel meta information.
             * Object is used to apply the grid and block extents and optionally the amount of dynamic shared memory to
             * the kernel.
             */
            template<typename T_KernelFunctor, typename T_WorkerCfg>
            struct KernelPreperationWrapper
            {
                using KernelFunctor = detail::LockStepKernel<T_KernelFunctor, T_WorkerCfg>;
                KernelFunctor const m_kernelFunctor;
                pmacc::exec::detail::KernelMetaData const m_metaData;

                HINLINE KernelPreperationWrapper(
                    T_KernelFunctor const& kernelFunctor,
                    std::string const& file = std::string(),
                    size_t const line = 0)
                    : m_kernelFunctor(kernelFunctor)
                    , m_metaData(file, line)
                {
                }

                /** Configured kernel object.
                 *
                 * This objects contains the functor and the starting parameter.
                 *
                 * @tparam T_VectorGrid type which defines the grid extents (type must be castable to cupla dim3)
                 * @tparam T_VectorBlock type which defines the block extents (type must be castable to cupla dim3)
                 *
                 * @param gridExtent grid extent configuration for the kernel
                 * @param blockExtent block extent configuration for the kernel
                 *
                 * @return object to bind arguments to a kernel
                 *
                 * @{
                 */
                template<typename T_VectorGrid>
                HINLINE auto operator()(T_VectorGrid const& gridExtent) const
                    -> pmacc::exec::detail::KernelLauncher<KernelFunctor>
                {
                    return {m_kernelFunctor, m_metaData, gridExtent, T_WorkerCfg::getNumWorkers()};
                }

                /**
                 * @param sharedMemByte dynamic shared memory used by the kernel (in byte)
                 */
                template<typename T_VectorGrid>
                HINLINE auto operator()(T_VectorGrid const& gridExtent, size_t const sharedMemByte) const
                    -> pmacc::exec::detail::KernelLauncher<pmacc::exec::detail::KernelWithDynSharedMem<KernelFunctor>>
                {
                    return {
                        pmacc::exec::detail::KernelWithDynSharedMem<KernelFunctor>(m_kernelFunctor, sharedMemByte),
                        m_metaData,
                        gridExtent,
                        T_WorkerCfg::getNumWorkers()};
                }
                /**@}*/
            };
        } // namespace detail

        /** Creates a kernel object.
         *
         * example for lambda usage:
         *
         * @code{.cpp}
         *   pmacc::lockstep::exec::kernel([]ALPAKA_FN_ACC(auto const& acc) -> void{
         *       printf("Hello World.\n");
         *   })(1,1)()
         * @endcode
         *
         * @tparam T_KernelFunctor type of the kernel functor
         * @param kernelFunctor instance of the functor, lambda are supported
         * @param file file name (for debug)
         * @param line line number in the file (for debug)
         */
        template<typename T_KernelFunctor, uint32_t T_numSuggestedWorkers>
        inline auto kernel(
            T_KernelFunctor const& kernelFunctor,
            WorkerCfg<T_numSuggestedWorkers> const& /*workerCfg*/,
            std::string const& file = std::string(),
            size_t const line = 0)
            -> detail::KernelPreperationWrapper<T_KernelFunctor, WorkerCfg<T_numSuggestedWorkers>>
        {
            return detail::KernelPreperationWrapper<T_KernelFunctor, WorkerCfg<T_numSuggestedWorkers>>(
                kernelFunctor,
                file,
                line);
        }

    } // namespace exec
} // namespace pmacc::lockstep

/** Create a kernel object out of a functor instance.
 *
 * This macro add the current filename and line number to the kernel object.
 * @see ::pmacc::lockstep::exec::kernel
 *
 * @param ... instance of kernel functor
 */
#define PMACC_LOCKSTEP_KERNEL(...)                                                                                    \
    ::pmacc::lockstep::exec::kernel(__VA_ARGS__, __FILE__, static_cast<size_t>(__LINE__))
