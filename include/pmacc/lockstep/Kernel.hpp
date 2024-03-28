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
#include "pmacc/exec/KernelLauncher.hpp"
#include "pmacc/exec/KernelMetaData.hpp"
#include "pmacc/exec/KernelWithDynSharedMem.hpp"
#include "pmacc/lockstep/BlockCfg.hpp"
#include "pmacc/lockstep/Worker.hpp"
#include "pmacc/types.hpp"

namespace pmacc::lockstep
{
    namespace exec
    {
        namespace detail
        {
            template<typename T_Kernel, typename T_BlockCfg>
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
                    auto const worker = lockstep::detail::BlockCfgAssume1DBlock::getWorker<T_BlockCfg>(acc);
                    T_Kernel::template operator()(worker, std::forward<T_Args>(args)...);
                }
            };


            /** Wraps a user kernel functor to prepare the execution on the device.
             *
             * This objects contains the kernel functor, kernel meta information.
             * Object is used to apply the grid and block extents and optionally the amount of dynamic shared memory to
             * the kernel.
             */
            template<typename T_UserKernelFunctor>
            struct KernelPreperationWrapper
            {
                template<typename T_BlockCfg>
                using KernelFunctor = detail::LockStepKernel<T_UserKernelFunctor, T_BlockCfg>;

                T_UserKernelFunctor const m_UserKernelFunctor;
                pmacc::exec::detail::KernelMetaData const m_metaData;

                HINLINE KernelPreperationWrapper(
                    T_UserKernelFunctor const& kernelFunctor,
                    std::string const& file = std::string(),
                    size_t const line = 0)
                    : m_UserKernelFunctor(kernelFunctor)
                    , m_metaData(file, line)
                {
                }

                /** Configured kernel object.
                 *
                 * @tparam T_VectorGrid type which defines the grid extents
                 * @tparam T_blockSize number of block indices
                 *
                 * @param gridSize grid extent configuration for the kernel
                 *
                 * @return object to bind arguments to a kernel
                 *
                 * @{
                 */
                template<uint32_t T_blockSize, typename T_VectorGrid>
                HINLINE auto config(T_VectorGrid const& gridSize) const
                {
                    static_assert(
                        lockstep::traits::hasBlockCfg_v<T_UserKernelFunctor> == false,
                        "You are not allowed to overwrite the block size defined by the kernel!");
                    auto blockCfg = lockstep::makeBlockCfg<T_blockSize>();
                    using BlockConfiguration = decltype(blockCfg);
                    constexpr uint32_t dim = pmacc::exec::detail::GetDim<T_VectorGrid>::dim;
                    auto blockExtent = DataSpace<dim>::create(1);
                    blockExtent.x() = BlockConfiguration::numWorkers();
                    return pmacc::exec::detail::KernelLauncher<KernelFunctor<BlockConfiguration>, dim>{
                        m_UserKernelFunctor,
                        m_metaData,
                        gridSize,
                        blockExtent};
                }

                /**
                 * The block size is derived from the kernel functor.
                 * The kernel must define the member type `BlockDomSizeND` or the compile time constance
                 * `blockDomSize`.
                 */
                template<typename T_VectorGrid>
                HINLINE auto config(T_VectorGrid const& gridSize) const
                {
                    static_assert(
                        lockstep::traits::hasBlockCfg_v<T_UserKernelFunctor> == true,
                        "Kernel must define the type BlockDomSizeND or the value blockDomSize!");
                    auto blockCfg = lockstep::makeBlockCfg(m_UserKernelFunctor);
                    using BlockConfiguration = decltype(blockCfg);
                    constexpr uint32_t dim = pmacc::exec::detail::GetDim<T_VectorGrid>::dim;
                    auto blockSizeND = DataSpace<dim>::create(1);
                    blockSizeND.x() = BlockConfiguration::numWorkers();
                    return pmacc::exec::detail::KernelLauncher<KernelFunctor<BlockConfiguration>, dim>{
                        m_UserKernelFunctor,
                        m_metaData,
                        gridSize,
                        blockSizeND};
                }

                /**
                 * @param blockSizeConcept Object which describe the number of required indices within the block.
                 *                         Object must be supported by lockstep::makeBlockCfg().
                 */
                template<typename T_VectorGrid, typename T_BlockSizeConcept>
                HINLINE auto config(T_VectorGrid const& gridSize, T_BlockSizeConcept const& blockSizeConcept) const
                {
                    static_assert(
                        lockstep::traits::hasBlockCfg_v<T_UserKernelFunctor> == false,
                        "You are not allowed to overwrite the block size defined by the kernel!");
                    auto blockCfg = lockstep::makeBlockCfg(blockSizeConcept);
                    using BlockConfiguration = decltype(blockCfg);
                    constexpr uint32_t dim = pmacc::exec::detail::GetDim<T_VectorGrid>::dim;

                    auto blockExtent = DataSpace<dim>::create(1);
                    blockExtent.x() = BlockConfiguration::numWorkers();
                    return pmacc::exec::detail::KernelLauncher<KernelFunctor<BlockConfiguration>, dim>{
                        m_UserKernelFunctor,
                        m_metaData,
                        gridSize,
                        blockExtent};
                }

                /**
                 * @param blockSize block extent configuration for the kernel
                 * @param sharedMemByte dynamic shared memory used by the kernel (in byte)
                 */
                template<uint32_t T_blockSize, typename T_VectorGrid>
                HINLINE auto configSMem(T_VectorGrid const& gridSize, size_t const sharedMemByte) const
                {
                    static_assert(
                        lockstep::traits::hasBlockCfg_v<T_UserKernelFunctor> == false,
                        "You are not allowed to overwrite the block size defined by the kernel!");
                    auto blockCfg = lockstep::makeBlockCfg<T_blockSize>();
                    using BlockConfiguration = decltype(blockCfg);
                    constexpr uint32_t dim = pmacc::exec::detail::GetDim<T_VectorGrid>::dim;
                    auto blockExtent = DataSpace<dim>::create(1);
                    blockExtent.x() = BlockConfiguration::numWorkers();
                    return pmacc::exec::detail::KernelLauncher<
                        pmacc::exec::detail::KernelWithDynSharedMem<KernelFunctor<BlockConfiguration>>,
                        dim>{
                        pmacc::exec::detail::KernelWithDynSharedMem<KernelFunctor<BlockConfiguration>>(
                            m_UserKernelFunctor,
                            sharedMemByte),
                        m_metaData,
                        gridSize,
                        blockExtent};
                }

                /**
                 * The block size is derived from the kernel functor.
                 * The kernel must define the member type `BlockDomSizeND` or the compile time constance
                 * `blockDomSize`.
                 *
                 * @param sharedMemByte dynamic shared memory used by the kernel (in byte)
                 */
                template<typename T_VectorGrid>
                HINLINE auto configSMem(T_VectorGrid const& gridSize, size_t const sharedMemByte) const
                {
                    static_assert(
                        lockstep::traits::hasBlockCfg_v<T_UserKernelFunctor> == true,
                        "Kernel must define the type BlockDomSizeND or the value blockDomSize!");
                    auto blockCfg = lockstep::makeBlockCfg(m_UserKernelFunctor);
                    using BlockConfiguration = decltype(blockCfg);
                    constexpr uint32_t dim = pmacc::exec::detail::GetDim<T_VectorGrid>::dim;
                    auto blockExtent = DataSpace<dim>::create(1);
                    blockExtent.x() = BlockConfiguration::numWorkers();
                    return pmacc::exec::detail::KernelLauncher<
                        pmacc::exec::detail::KernelWithDynSharedMem<KernelFunctor<BlockConfiguration>>,
                        dim>{
                        pmacc::exec::detail::KernelWithDynSharedMem<KernelFunctor<BlockConfiguration>>(
                            m_UserKernelFunctor,
                            sharedMemByte),
                        m_metaData,
                        gridSize,
                        blockExtent};
                }

                /**
                 * @param blockSizeConcept Object which describe the number of required indices within the block.
                 *                         Object must be supported by lockstep::makeBlockCfg().
                 * @param sharedMemByte dynamic shared memory used by the kernel (in byte)
                 */
                template<typename T_VectorGrid, typename T_BlockSizeConcept>
                HINLINE auto configSMem(
                    T_VectorGrid const& gridSize,
                    T_BlockSizeConcept const& blockSizeConcept,
                    size_t const sharedMemByte) const
                {
                    static_assert(
                        lockstep::traits::hasBlockCfg_v<T_UserKernelFunctor> == false,
                        "You are not allowed to overwrite the block size defined by the kernel!");
                    auto blockCfg = lockstep::makeBlockCfg(blockSizeConcept);
                    using BlockConfiguration = decltype(blockCfg);
                    constexpr uint32_t dim = pmacc::exec::detail::GetDim<T_VectorGrid>::dim;

                    auto blockExtent = DataSpace<dim>::create(1);
                    blockExtent.x() = BlockConfiguration::numWorkers();
                    return pmacc::exec::detail::KernelLauncher<
                        pmacc::exec::detail::KernelWithDynSharedMem<KernelFunctor<BlockConfiguration>>,
                        dim>{
                        pmacc::exec::detail::KernelWithDynSharedMem<KernelFunctor<BlockConfiguration>>(
                            m_UserKernelFunctor,
                            sharedMemByte),
                        m_metaData,
                        gridSize,
                        blockExtent};
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
        template<typename T_KernelFunctor>
        inline auto kernel(
            T_KernelFunctor const& kernelFunctor,
            std::string const& file = std::string(),
            size_t const line = 0) -> detail::KernelPreperationWrapper<T_KernelFunctor>
        {
            return detail::KernelPreperationWrapper<T_KernelFunctor>(kernelFunctor, file, line);
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
