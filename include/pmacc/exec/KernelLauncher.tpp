/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once


#include "pmacc/Environment.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/exec/KernelLauncher.hpp"
#include "pmacc/traits/GetNComponents.hpp"
#include "pmacc/types.hpp"

#include <string>
#include <typeinfo>


/* No namespace in this file since we only declare macro defines */

/*if this flag is defined all kernel calls would be checked and synchronize
 * this flag must set by the compiler or inside of the Makefile
 */
#if (PMACC_SYNC_KERNEL == 1)
#    define PMACC_CHECK_KERNEL_MSG(...) PMACC_CHECK_ALPAKA_CALL_MSG(__VA_ARGS__)
#else
/*no synchronize and check of kernel calls*/
#    define PMACC_CHECK_KERNEL_MSG(...) ;
#endif


namespace pmacc::exec::detail
{
    template<typename T_Kernel, uint32_t T_dim>
    struct KernelLauncher
    {
        //! kernel functor
        T_Kernel const m_kernel;
        std::string const m_file;
        size_t const m_line;
        //! grid extents for the kernel
        math::Vector<IdxType, T_dim> const m_gridExtent;
        //! block extents for the kernel
        math::Vector<IdxType, T_dim> const m_blockExtent;

        /** kernel starter object
         *
         * @param kernel pmacc Kernel
         */
        template<typename T_VectorGrid, typename T_VectorBlock>
        HINLINE KernelLauncher(
            T_Kernel const& kernel,
            std::string const& file,
            size_t const line,
            T_VectorGrid const& gridExtent,
            T_VectorBlock const& blockExtent)
            : m_kernel(kernel)
            , m_file(file)
            , m_line(line)
            , m_gridExtent(gridExtent)
            , m_blockExtent(blockExtent)
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
            std::string const kernelInfo = kernelName + std::string(" [") + m_file + std::string(":")
                                           + std::to_string(m_line) + std::string(" ]");

            PMACC_CHECK_KERNEL_MSG(
                alpaka::wait(manager::Device<ComputeDevice>::get().current());
                , std::string("Crash before kernel call ") + kernelInfo);

            pmacc::TaskKernel* taskKernel = pmacc::Environment<>::get().Factory().createTaskKernel(kernelName);

            auto gridExtent = m_gridExtent.toAlpakaKernelVec();
            auto blockExtent = m_blockExtent.toAlpakaKernelVec();
            auto elemExtent = math::Vector<IdxType, T_dim>::create(1).toAlpakaKernelVec();
            auto workDiv
                = ::alpaka::WorkDivMembers<::alpaka::DimInt<T_dim>, IdxType>(gridExtent, blockExtent, elemExtent);

            auto const kernelTask
                = ::alpaka::createTaskKernel<Acc<T_dim>>(workDiv, m_kernel, std::forward<T_Args>(args)...);

            auto queue = taskKernel->getAlpakaQueue();

            ::alpaka::enqueue(queue, kernelTask);

            PMACC_CHECK_KERNEL_MSG(
                alpaka::wait(manager::Device<ComputeDevice>::get().current());
                , std::string("Crash after kernel launch ") + kernelInfo);
            taskKernel->activateChecks();
            PMACC_CHECK_KERNEL_MSG(
                alpaka::wait(manager::Device<ComputeDevice>::get().current());
                , std::string("Crash after kernel activation") + kernelInfo);
        }
    };

} // namespace pmacc::exec::detail
