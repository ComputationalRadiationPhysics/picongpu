/* Copyright 2013-2023 Felix Schmitt, Rene Widera, Benjamin Worpitz
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


#include "pmacc/Environment.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/exec/KernelLauncher.hpp"
#include "pmacc/exec/KernelMetaData.hpp"
#include "pmacc/exec/KernelWithDynSharedMem.hpp"
#include "pmacc/traits/GetNComponents.hpp"
#include "pmacc/types.hpp"

#include <string>


namespace pmacc::exec
{
    namespace detail
    {
        /** Wraps a user kernel functor to prepare the execution on the device.
         *
         * This objects contains the kernel functor, kernel meta information.
         * Object is used to apply the grid and block extents and optionally the amount of dynamic shared memory to the
         * kernel.
         */
        template<typename T_KernelFunctor>
        struct KernelPreperationWrapper
        {
            T_KernelFunctor const m_kernelFunctor;
            KernelMetaData const m_metaData;

            HINLINE KernelPreperationWrapper(
                T_KernelFunctor const& kernelFunctor,
                std::string const& file = std::string(),
                size_t const line = 0)
                : m_kernelFunctor(kernelFunctor)
                , m_metaData(file, line)
            {
            }

            /** Apply grid and block extents and optionally dynamic shared memory to the wrapped functor.
             *
             * @tparam T_VectorGrid type which defines the grid extents (type must be cast-able to cupla dim3)
             * @tparam T_VectorBlock type which defines the block extents (type must be cast-able to cupla dim3)
             *
             * @param gridExtent grid extent configuration for the kernel
             * @param blockExtent block extent configuration for the kernel
             *
             * @return object with user kernel functor and launch parameters
             *
             * @{
             */
            template<typename T_VectorGrid, typename T_VectorBlock>
            HINLINE auto operator()(T_VectorGrid const& gridExtent, T_VectorBlock const& blockExtent) const
                -> KernelLauncher<T_KernelFunctor>;

            /**
             * @param sharedMemByte dynamic shared memory used by the kernel (in byte)
             */
            template<typename T_VectorGrid, typename T_VectorBlock>
            HINLINE auto operator()(
                T_VectorGrid const& gridExtent,
                T_VectorBlock const& blockExtent,
                size_t const sharedMemByte) const -> KernelLauncher<KernelWithDynSharedMem<T_KernelFunctor>>;
            /**@}*/
        };
    } // namespace detail

    /** Creates a kernel object.
     *
     * example for lambda usage:
     *
     * @code{.cpp}
     *   pmacc::exec::kernel([]ALPAKA_FN_ACC(auto const& acc) -> void{
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
    auto kernel(T_KernelFunctor const& kernelFunctor, std::string const& file = std::string(), size_t const line = 0)
        -> detail::KernelPreperationWrapper<T_KernelFunctor>
    {
        return detail::KernelPreperationWrapper<T_KernelFunctor>(kernelFunctor, file, line);
    }
} // namespace pmacc::exec


namespace alpaka
{
    namespace trait
    {
        /** alpaka trait specialization to define dynamic shared memory for a kernel.
         *
         * All PMacc kernel with dynamic shared memory usage are wrapped by KernelWithDynSharedMem where the required
         * amount of shared memory is available as member variable.
         */
        template<typename T_UserKernel, typename T_Acc>
        struct BlockSharedMemDynSizeBytes<::pmacc::exec::detail::KernelWithDynSharedMem<T_UserKernel>, T_Acc>
        {
            template<typename... TArgs>
            ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
                ::pmacc::exec::detail::KernelWithDynSharedMem<T_UserKernel> const& userKernel,
                TArgs&&...) -> ::alpaka::Idx<T_Acc>
            {
                return userKernel.m_dynSharedMemBytes;
            }
        };
    } // namespace trait
} // namespace alpaka

/** Create a kernel object out of a functor instance.
 *
 * This macro add the current filename and line number to the kernel object.
 * @see ::pmacc::exec::kernel
 *
 * @param ... instance of kernel functor
 */
#define PMACC_KERNEL(...) ::pmacc::exec::kernel(__VA_ARGS__, __FILE__, static_cast<size_t>(__LINE__))


#include "pmacc/eventSystem/events/kernelEvents.tpp"
#include "pmacc/exec/KernelLauncher.tpp"
