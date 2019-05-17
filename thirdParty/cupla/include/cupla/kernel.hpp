/* Copyright 2016 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include "cupla/namespace.hpp"
#include "cupla/types.hpp"

#include "cupla/datatypes/dim3.hpp"
#include "cupla/datatypes/uint.hpp"
#include "cupla/manager/Stream.hpp"
#include "cupla/manager/Device.hpp"
#include "cupla/traits/IsThreadSeqAcc.hpp"

#include <utility>

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{

    /** get block and elements extents
     *
     * can swap the block and element extents depend on the selected Alpaka
     * accelerator
     */
    template<
        typename T_Acc,
        bool T_isThreadSeqAcc = traits::IsThreadSeqAcc< T_Acc >::value
    >
    struct GetBlockAndElemExtents
    {
        static void get( dim3 const & , dim3 const &  )
        { }
    };

    template< typename T_Acc >
    struct GetBlockAndElemExtents<
        T_Acc,
        true
    >
    {
        static void get( dim3 & blockSize, dim3 & elemSize )
        {
            std::swap( blockSize, elemSize );
        }
    };

    struct KernelHelper
    {
        static cuplaStream_t
        getStream(
        size_t /*sharedMemSize*/ = 0,
            cuplaStream_t stream = 0
        )
        {
            return stream;
        }

        static size_t
        getSharedMemSize(
            size_t sharedMemSize = 0,
        cuplaStream_t /*stream*/ = 0
        )
        {
            return sharedMemSize;
        }

    };

    template<
        typename T_Kernel
    >
    struct CuplaKernel :
        public T_Kernel
    {
        size_t const  m_dynSharedMemBytes;

        CuplaKernel( size_t const & dynSharedMemBytes ) :
            m_dynSharedMemBytes( dynSharedMemBytes )
        { }
    };

    template<
        typename T_Acc,
        typename T_Kernel,
        typename T_Stream,
        typename... T_Args
    >
    void startKernel(
        T_Kernel const & kernel,
        uint3 const & gridSize,
        uint3 const & blockSize,
        uint3 const & elemPerThread,
        T_Stream & stream,
        T_Args && ... args
    ){
        auto dev( manager::Device<AccDev>::get().current() );
        ::alpaka::workdiv::WorkDivMembers<
          KernelDim,
          IdxType
        > workDiv(
            static_cast<IdxVec3>(gridSize),
            static_cast<IdxVec3>(blockSize),
            static_cast<IdxVec3>(elemPerThread)
        );
        auto const exec(::alpaka::kernel::createTaskKernel<T_Acc>(workDiv, kernel, args...));
        ::alpaka::queue::enqueue(stream, exec);
    }

} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla


namespace alpaka
{
namespace kernel
{
namespace traits
{
    //! CuplaKernel has defined the extern shared memory as member
    template<
        typename T_UserKernel,
        typename T_Acc
    >
    struct BlockSharedMemDynSizeBytes<
        ::cupla::CuplaKernel< T_UserKernel >,
        T_Acc
    >
    {
        template<
            typename... TArgs
        >
        ALPAKA_FN_HOST_ACC
        static auto
        getBlockSharedMemDynSizeBytes(
            ::cupla::CuplaKernel< T_UserKernel > const & userKernel,
            TArgs const & ...)
        -> ::alpaka::idx::Idx<T_Acc>
        {
            return userKernel.m_dynSharedMemBytes;
        }
    };
} // namespace traits
} // namespace kernel
} // namespace alpaka



#define CUPLA_CUDA_KERNEL_PARAMS(...)                                          \
    const KernelType cuplaTheOneAndOnlyKernel( cuplaSharedMemSize );           \
    cupla::startKernel<CuplaAcc>(                                              \
        cuplaTheOneAndOnlyKernel,                                              \
        cuplaGridSize,                                                         \
        cuplaBlockSize,                                                        \
        cuplaElemPerThread,                                                    \
        cuplaStream,                                                           \
        __VA_ARGS__                                                            \
    );                                                                         \
    }

#define CUPLA_CUDA_KERNEL_CONFIG(gridSize,blockSize,elemSize,...)              \
    const uint3 cuplaGridSize = dim3(gridSize);                                \
    const uint3 cuplaBlockSize = dim3(blockSize);                              \
    const uint3 cuplaElemPerThread = dim3(elemSize);                           \
    auto& cuplaStream(                                                         \
        cupla::manager::Stream<                                                \
            cupla::AccDev,                                                     \
            cupla::AccStream                                                   \
        >::get().stream(                                                       \
            cupla::KernelHelper::getStream( __VA_ARGS__ )                      \
        )                                                                      \
    );                                                                         \
    size_t const cuplaSharedMemSize = cupla::KernelHelper::getSharedMemSize(   \
        __VA_ARGS__                                                            \
    );                                                                         \
    CUPLA_CUDA_KERNEL_PARAMS

#define CUPLA_CUDA_KERNEL_CONFIG_DEFAULT(gridSize,blockSize,...)               \
    CUPLA_CUDA_KERNEL_CONFIG(gridSize,blockSize,dim3(),__VA_ARGS__)

/** default cupla kernel call
 *
 * The alpaka element level is ignored and always set to dim3(1,1,1)
 */
#define CUPLA_KERNEL(...) {                                                    \
    using CuplaAcc = cupla::Acc;                                               \
    using KernelType = ::cupla::CuplaKernel< __VA_ARGS__ >;                    \
    CUPLA_CUDA_KERNEL_CONFIG_DEFAULT

#define CUPLA_CUDA_KERNEL_CONFIG_OPTI(gridSize,blockSize,...)                  \
    dim3 tmp_cuplaBlockSize = dim3( blockSize );                               \
    dim3 tmp_cuplaElemSize;                                                    \
    cupla::GetBlockAndElemExtents<CuplaAcc>::get( tmp_cuplaBlockSize, tmp_cuplaElemSize );    \
    CUPLA_CUDA_KERNEL_CONFIG(gridSize,tmp_cuplaBlockSize,tmp_cuplaElemSize,__VA_ARGS__)

/** call the kernel with an hidden element layer
 *
 * The kernel must support the alpaka element level
 *
 * This kernel call swap the blockSize and the elemSize depending
 * on the activated accelerator.
 * This mean that in some devices the blockSize is set to one ( dim3(1,1,1) )
 * and the elemSize is set to the user defined blockSize
 */
#define CUPLA_KERNEL_OPTI(...) {                                               \
    using CuplaAcc = cupla::AccThreadSeq;                                      \
    using KernelType = ::cupla::CuplaKernel< __VA_ARGS__ >;                    \
    CUPLA_CUDA_KERNEL_CONFIG_OPTI

/** cupla kernel call with elements
 *
 * The kernel must support the alpaka element level
 */
#define CUPLA_KERNEL_ELEM(...) {                                               \
    using CuplaAcc = cupla::Acc;                                               \
    using KernelType = ::cupla::CuplaKernel< __VA_ARGS__ >;                    \
    CUPLA_CUDA_KERNEL_CONFIG

