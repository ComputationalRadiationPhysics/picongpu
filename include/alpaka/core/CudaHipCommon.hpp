/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Bernhard Manfred Gruber,
                  Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/elem/Traits.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/meta/Concatenate.hpp"
#include "alpaka/meta/TypeListOps.hpp"
#include "alpaka/offset/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <tuple>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#        include <cuda.h>
#        include <cuda_runtime.h>
#    endif

#    ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#        include <hip/hip_runtime.h>
#    endif

namespace alpaka
{
    namespace detail
    {
        using CudaHipBuiltinTypes1 = std::
            tuple<char1, double1, float1, int1, long1, longlong1, short1, uchar1, uint1, ulong1, ulonglong1, ushort1>;
        using CudaHipBuiltinTypes2 = std::
            tuple<char2, double2, float2, int2, long2, longlong2, short2, uchar2, uint2, ulong2, ulonglong2, ushort2>;
        using CudaHipBuiltinTypes3 = std::tuple<
            char3,
            dim3,
            double3,
            float3,
            int3,
            long3,
            longlong3,
            short3,
            uchar3,
            uint3,
            ulong3,
            ulonglong3,
            ushort3
// CUDA built-in variables have special types in clang native CUDA compilation
// defined in cuda_builtin_vars.h
#    if BOOST_COMP_CLANG_CUDA
            ,
            __cuda_builtin_threadIdx_t,
            __cuda_builtin_blockIdx_t,
            __cuda_builtin_blockDim_t,
            __cuda_builtin_gridDim_t
#    endif
            >;
        using CudaHipBuiltinTypes4 = std::
            tuple<char4, double4, float4, int4, long4, longlong4, short4, uchar4, uint4, ulong4, ulonglong4, ushort4>;
        using CudaHipBuiltinTypes = meta::
            Concatenate<CudaHipBuiltinTypes1, CudaHipBuiltinTypes2, CudaHipBuiltinTypes3, CudaHipBuiltinTypes4>;

        template<typename T>
        inline constexpr auto isCudaHipBuiltInType = meta::Contains<CudaHipBuiltinTypes, T>::value;
    } // namespace detail

#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    namespace cuda::trait
    {
        template<typename T>
        inline constexpr auto isCudaBuiltInType = alpaka::detail::isCudaHipBuiltInType<T>;
    } // namespace cuda::trait
#    endif

#    ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    namespace hip::trait
    {
        template<typename T>
        inline constexpr auto isHipBuiltInType = alpaka::detail::isCudaHipBuiltInType<T>;
    } // namespace hip::trait
#    endif

    namespace trait
    {
        //! The CUDA/HIP vectors 1D dimension get trait specialization.
        template<typename T>
        struct DimType<T, std::enable_if_t<meta::Contains<alpaka::detail::CudaHipBuiltinTypes1, T>::value>>
        {
            using type = DimInt<1u>;
        };

        //! The CUDA/HIP vectors 2D dimension get trait specialization.
        template<typename T>
        struct DimType<T, std::enable_if_t<meta::Contains<alpaka::detail::CudaHipBuiltinTypes2, T>::value>>
        {
            using type = DimInt<2u>;
        };

        //! The CUDA/HIP vectors 3D dimension get trait specialization.
        template<typename T>
        struct DimType<T, std::enable_if_t<meta::Contains<alpaka::detail::CudaHipBuiltinTypes3, T>::value>>
        {
            using type = DimInt<3u>;
        };

        //! The CUDA/HIP vectors 4D dimension get trait specialization.
        template<typename T>
        struct DimType<T, std::enable_if_t<meta::Contains<alpaka::detail::CudaHipBuiltinTypes4, T>::value>>
        {
            using type = DimInt<4u>;
        };

        //! The CUDA/HIP vectors elem type trait specialization.
        template<typename T>
        struct ElemType<T, std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<T>>>
        {
            using type = decltype(std::declval<T>().x);
        };

        template<typename TCudaHipBuiltin>
        struct GetExtents<TCudaHipBuiltin, std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TCudaHipBuiltin>>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC auto operator()(TCudaHipBuiltin const& value) const
                -> Vec<Dim<TCudaHipBuiltin>, Idx<TCudaHipBuiltin>>
            {
                constexpr auto dim = Dim<TCudaHipBuiltin>::value;
                if constexpr(dim == 1)
                    return {value.x};
                else if constexpr(dim == 2)
                    return {value.y, value.x};
                else if constexpr(dim == 3)
                    return {value.z, value.y, value.x};
                else if constexpr(dim == 4)
                    return {value.w, value.z, value.y, value.x};
                else
                    static_assert(sizeof(value) == 0, "Not implemented");

                ALPAKA_UNREACHABLE({});
            }
        };

        template<typename TCudaHipBuiltin>
        struct GetOffsets<TCudaHipBuiltin, std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TCudaHipBuiltin>>>
            : GetExtents<TCudaHipBuiltin>
        {
        };

        //! The CUDA/HIP vectors idx type trait specialization.
        template<typename TIdx>
        struct IdxType<TIdx, std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TIdx>>>
        {
            using type = std::size_t;
        };
    } // namespace trait
} // namespace alpaka

#endif
