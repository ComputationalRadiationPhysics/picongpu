/* Copyright 2013-2022 Felix Schmitt, Rene Widera, Benjamin Worpitz
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
#include "pmacc/traits/GetNComponents.hpp"
#include "pmacc/types.hpp"

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


namespace pmacc::exec
{
    namespace detail
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

        //! Meta data of a device kernel
        class KernelMetaData
        {
            //! file name
            std::string const m_file;
            //! line number
            size_t const m_line;

        public:
            KernelMetaData(std::string const& file, size_t const line) : m_file(file), m_line(line)
            {
            }

            //! file name from where the kernel is called
            std::string getFile() const
            {
                return m_file;
            }

            //! line number in the file where the kernel is called
            size_t getLine() const
            {
                return m_line;
            }
        };


        /** Object to launch the kernel functor on the device.
         *
         * This objects contains the kernel functor, kernel meta information and the launch parameters.
         * Object is used to enqueue the kernel with user arguments on the device.
         *
         * @tparam T_Kernel pmacc Kernel object
         */
        template<typename T_Kernel>
        struct KernelLauncher;

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


        template<typename T_Kernel>
        struct KernelLauncher
        {
            //! kernel functor
            T_Kernel const m_kernel;
            //! Debug meta data for the kernel functor.
            KernelMetaData const m_metaData;
            //! grid extents for the kernel
            cupla::dim3 const m_gridExtent;
            //! block extents for the kernel
            cupla::dim3 const m_blockExtent;

            /** kernel starter object
             *
             * @param kernel pmacc Kernel
             */
            template<typename T_VectorGrid, typename T_VectorBlock>
            HINLINE KernelLauncher(
                T_Kernel const& kernel,
                detail::KernelMetaData const& kernelMetaData,
                T_VectorGrid const& gridExtent,
                T_VectorBlock const& blockExtent)
                : m_kernel(kernel)
                , m_metaData(kernelMetaData)
                , m_gridExtent(DataSpace<traits::GetNComponents<T_VectorGrid>::value>(gridExtent).toDim3())
                , m_blockExtent(DataSpace<traits::GetNComponents<T_VectorBlock>::value>(blockExtent).toDim3())
            {
            }

            /** Enqueue the kernel functor with the given arguments for execution.
             *
             * The stream into which the kernel is enqueued is automatically chosen by PMacc's event system.
             *
             * @tparam T_Args types of the arguments
             * @param args arguments for the kernel functor
             */
            template<typename... T_Args>
            HINLINE void operator()(T_Args&&... args) const
            {
                std::string const kernelName = typeid(m_kernel).name();
                std::string const kernelInfo = kernelName + std::string(" [") + m_metaData.getFile() + std::string(":")
                    + std::to_string(m_metaData.getLine()) + std::string(" ]");

                CUDA_CHECK_KERNEL_MSG(cuplaDeviceSynchronize(), std::string("Crash before kernel call ") + kernelInfo);

                pmacc::TaskKernel* taskKernel = pmacc::Environment<>::get().Factory().createTaskKernel(kernelName);

                /* The next 3 lines cast pmacc vectors or integral values used for the execution extents into alpaka
                 * vectors.
                 * Note to be cupla compatible we use currently always 3-dimensional kernels.
                 *
                 * @todo find a better way to write this part of code, in general alpaka understands PMacc vectors but
                 * PMacc has no easy way t cast integral numbers to an 3-dimensional vector.
                 */
                auto gridExtent = cupla::IdxVec3(m_gridExtent.z, m_gridExtent.y, m_gridExtent.x);
                auto blockExtent = cupla::IdxVec3(m_blockExtent.z, m_blockExtent.y, m_blockExtent.x);
                auto elemExtent = cupla::IdxVec3::ones();
                auto workDiv
                    = ::alpaka::WorkDivMembers<cupla::KernelDim, cupla::IdxType>(gridExtent, blockExtent, elemExtent);

                auto const kernelTask
                    = ::alpaka::createTaskKernel<cupla::Acc>(workDiv, m_kernel, std::forward<T_Args>(args)...);

                auto cuplaStream = taskKernel->getCudaStream();
                auto& stream = cupla::manager::Stream<cupla::AccDev, cupla::AccStream>::get().stream(cuplaStream);

                ::alpaka::enqueue(stream, kernelTask);

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
