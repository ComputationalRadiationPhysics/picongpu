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

    /** wrapper for kernel types
     *
     * This implements the possibility to define dynamic shared memory without
     * specializing the needed alpaka trait BlockSharedMemDynSizeBytes
     */
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

    /** execute a kernel
     *
     * @tparam T_KernelType type of the kernel
     * @tparam T_Acc accelerator used to execute the kernel
     *
     */
    template<
        typename T_KernelType,
        typename T_Acc
    >
    class KernelExecutor
    {
        IdxVec3 const m_gridSize;
        IdxVec3 const m_blockSize;
        IdxVec3 const m_elemSize;
        uint32_t const m_dynSharedMemSize;
        cuplaStream_t const m_stream;

    public:
        KernelExecutor(
            dim3 const & gridSize,
            dim3 const & blockSize,
            dim3 const & elemSize,
            uint32_t const & dynSharedMemSize,
            cuplaStream_t const & stream
        ) :
            m_gridSize( gridSize.z, gridSize.y, gridSize.x ),
            m_blockSize( blockSize.z, blockSize.y, blockSize.x ),
            m_elemSize( elemSize.z, elemSize.y, elemSize.x ),
            m_dynSharedMemSize( dynSharedMemSize ),
            m_stream( stream )
        {}

        template< typename... T_Args >
        void operator()( T_Args && ... args ) const
        {
            ::alpaka::WorkDivMembers<
              KernelDim,
              IdxType
            > workDiv(
                m_gridSize,
                m_blockSize,
                m_elemSize
            );
            auto const exec(
                ::alpaka::createTaskKernel< T_Acc >(
                    workDiv,
                    CuplaKernel< T_KernelType >{ m_dynSharedMemSize },
                    std::forward< T_Args >( args )...
                )
            );
            auto & stream = cupla::manager::Stream<
                cupla::AccDev,
                cupla::AccStream
            >::get().stream( m_stream );

            ::alpaka::enqueue(stream, exec);
        }
    };

    /** Cuda like configuration interface for a kernel
     *
     * Interface is compatible to the argument order of a cuda kernel `T_KernelType<<<...>>>`
     */
    template<
        typename T_KernelType
    >
    struct KernelCudaLike
    {
        auto operator()(
            dim3 const & gridSize,
            dim3 const & blockSize,
            uint32_t const & dynSharedMemSize = 0,
            cuplaStream_t const & stream = 0
        ) const
        -> KernelExecutor<
            T_KernelType,
            cupla::Acc
        >
        {
            return KernelExecutor<
                T_KernelType,
                cupla::Acc
            >(gridSize, blockSize, dim3(), dynSharedMemSize, stream);
        }
    };

    /* Kernel configuration interface with element support
     *
     * The kernel must support the alpaka element layer.
     *
     * Swap the blockSize and the elemSize depending on the activated accelerator.
     * This mean that in some devices the blockSize is set to one ( dim3(1,1,1) )
     * and the elemSize is set to the user defined blockSize
     */
    template<
        typename T_KernelType
    >
    struct SwitchableElementLevel
    {
        auto operator()(
            dim3 const & gridSize,
            dim3 const & blockSize,
            uint32_t const & dynSharedMemSize = 0,
            cuplaStream_t const & stream = 0
        ) const
        -> KernelExecutor<
            T_KernelType,
            cupla::AccThreadSeq
        >
        {
            dim3 tmpBlockSize = blockSize;
            dim3 tmpElemSize;
            GetBlockAndElemExtents<cupla::AccThreadSeq>::get( tmpBlockSize, tmpElemSize );

            return KernelExecutor<
                T_KernelType,
                cupla::AccThreadSeq
            >(gridSize, tmpBlockSize, tmpElemSize, dynSharedMemSize, stream);
        }
    };

    /** Kernel configuration interface with element support
     *
     * The kernel must support the alpaka element level
     */
    template<
        typename T_KernelType
    >
    struct KernelWithElementLevel
    {
        auto operator()(
            dim3 const & gridSize,
            dim3 const & blockSize,
            dim3 const & elemSize,
            uint32_t const & dynSharedMemSize = 0,
            cuplaStream_t const & stream = 0
        )  const
        -> KernelExecutor<
            T_KernelType,
            cupla::Acc
        >
        {
            return KernelExecutor<
                T_KernelType,
                cupla::Acc
            >(gridSize, blockSize, elemSize, dynSharedMemSize, stream);
        }
    };

} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla


namespace alpaka
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
        -> ::alpaka::Idx<T_Acc>
        {
            return userKernel.m_dynSharedMemBytes;
        }
    };
} // namespace traits
} // namespace alpaka


/** default cupla kernel call
 *
 * The alpaka element level is ignored and always set to dim3(1,1,1)
 */
#define CUPLA_KERNEL(...) ::cupla::KernelCudaLike<__VA_ARGS__>{}

/** call the kernel with an hidden element layer
 *
 * The kernel must support the alpaka element level
 *
 * This kernel call swap the blockSize and the elemSize depending
 * on the activated accelerator.
 * This mean that in some devices the blockSize is set to one ( dim3(1,1,1) )
 * and the elemSize is set to the user defined blockSize
 */
#define CUPLA_KERNEL_OPTI(...) ::cupla::SwitchableElementLevel<__VA_ARGS__>{}

/** cupla kernel call with elements
 *
 * The kernel must support the alpaka element level
 */
#define CUPLA_KERNEL_ELEM(...) ::cupla::KernelWithElementLevel<__VA_ARGS__>{}
