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
#include "pmacc/exec/KernelLauncher.hpp"
#include "pmacc/exec/KernelMetaData.hpp"
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


namespace pmacc::exec::detail
{
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

            CUDA_CHECK_KERNEL_MSG(cuplaGetLastError(), std::string("Last error after kernel launch ") + kernelInfo);
            CUDA_CHECK_KERNEL_MSG(cuplaDeviceSynchronize(), std::string("Crash after kernel launch ") + kernelInfo);
            taskKernel->activateChecks();
            CUDA_CHECK_KERNEL_MSG(cuplaDeviceSynchronize(), std::string("Crash after kernel activation") + kernelInfo);
        }
    };

} // namespace pmacc::exec::detail
