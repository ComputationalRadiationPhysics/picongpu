/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/elem/Traits.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/meta/Concatenate.hpp>
#    include <alpaka/meta/TypeListOps.hpp>
#    include <alpaka/offset/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <tuple>

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

        //! The CUDA/HIP vectors extent get trait specialization.
        template<typename TExtent>
        struct GetExtent<
            DimInt<Dim<TExtent>::value - 1u>,
            TExtent,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TExtent> && (Dim<TExtent>::value >= 1)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getExtent(TExtent const& extent)
            {
                return extent.x;
            }
        };
        //! The CUDA/HIP vectors extent get trait specialization.
        template<typename TExtent>
        struct GetExtent<
            DimInt<Dim<TExtent>::value - 2u>,
            TExtent,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TExtent> && (Dim<TExtent>::value >= 2)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getExtent(TExtent const& extent)
            {
                return extent.y;
            }
        };
        //! The CUDA/HIP vectors extent get trait specialization.
        template<typename TExtent>
        struct GetExtent<
            DimInt<Dim<TExtent>::value - 3u>,
            TExtent,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TExtent> && (Dim<TExtent>::value >= 3)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getExtent(TExtent const& extent)
            {
                return extent.z;
            }
        };
        //! The CUDA/HIP vectors extent get trait specialization.
        template<typename TExtent>
        struct GetExtent<
            DimInt<Dim<TExtent>::value - 4u>,
            TExtent,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TExtent> && (Dim<TExtent>::value >= 4)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getExtent(TExtent const& extent)
            {
                return extent.w;
            }
        };
        //! The CUDA/HIP vectors extent set trait specialization.
        template<typename TExtent, typename TExtentVal>
        struct SetExtent<
            DimInt<Dim<TExtent>::value - 1u>,
            TExtent,
            TExtentVal,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TExtent> && (Dim<TExtent>::value >= 1)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setExtent(TExtent const& extent, TExtentVal const& extentVal) -> void
            {
                extent.x = extentVal;
            }
        };
        //! The CUDA/HIP vectors extent set trait specialization.
        template<typename TExtent, typename TExtentVal>
        struct SetExtent<
            DimInt<Dim<TExtent>::value - 2u>,
            TExtent,
            TExtentVal,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TExtent> && (Dim<TExtent>::value >= 2)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setExtent(TExtent const& extent, TExtentVal const& extentVal) -> void
            {
                extent.y = extentVal;
            }
        };
        //! The CUDA/HIP vectors extent set trait specialization.
        template<typename TExtent, typename TExtentVal>
        struct SetExtent<
            DimInt<Dim<TExtent>::value - 3u>,
            TExtent,
            TExtentVal,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TExtent> && (Dim<TExtent>::value >= 3)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setExtent(TExtent const& extent, TExtentVal const& extentVal) -> void
            {
                extent.z = extentVal;
            }
        };
        //! The CUDA/HIP vectors extent set trait specialization.
        template<typename TExtent, typename TExtentVal>
        struct SetExtent<
            DimInt<Dim<TExtent>::value - 4u>,
            TExtent,
            TExtentVal,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TExtent> && (Dim<TExtent>::value >= 4)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setExtent(TExtent const& extent, TExtentVal const& extentVal) -> void
            {
                extent.w = extentVal;
            }
        };

        //! The CUDA/HIP vectors offset get trait specialization.
        template<typename TOffsets>
        struct GetOffset<
            DimInt<Dim<TOffsets>::value - 1u>,
            TOffsets,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TOffsets> && (Dim<TOffsets>::value >= 1)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const& offsets)
            {
                return offsets.x;
            }
        };
        //! The CUDA/HIP vectors offset get trait specialization.
        template<typename TOffsets>
        struct GetOffset<
            DimInt<Dim<TOffsets>::value - 2u>,
            TOffsets,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TOffsets> && (Dim<TOffsets>::value >= 2)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const& offsets)
            {
                return offsets.y;
            }
        };
        //! The CUDA/HIP vectors offset get trait specialization.
        template<typename TOffsets>
        struct GetOffset<
            DimInt<Dim<TOffsets>::value - 3u>,
            TOffsets,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TOffsets> && (Dim<TOffsets>::value >= 3)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const& offsets)
            {
                return offsets.z;
            }
        };
        //! The CUDA/HIP vectors offset get trait specialization.
        template<typename TOffsets>
        struct GetOffset<
            DimInt<Dim<TOffsets>::value - 4u>,
            TOffsets,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TOffsets> && (Dim<TOffsets>::value >= 4)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getOffset(TOffsets const& offsets)
            {
                return offsets.w;
            }
        };
        //! The CUDA/HIP vectors offset set trait specialization.
        template<typename TOffsets, typename TOffset>
        struct SetOffset<
            DimInt<Dim<TOffsets>::value - 1u>,
            TOffsets,
            TOffset,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TOffsets> && (Dim<TOffsets>::value >= 1)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(TOffsets const& offsets, TOffset const& offset) -> void
            {
                offsets.x = offset;
            }
        };
        //! The CUDA/HIP vectors offset set trait specialization.
        template<typename TOffsets, typename TOffset>
        struct SetOffset<
            DimInt<Dim<TOffsets>::value - 2u>,
            TOffsets,
            TOffset,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TOffsets> && (Dim<TOffsets>::value >= 2)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(TOffsets const& offsets, TOffset const& offset) -> void
            {
                offsets.y = offset;
            }
        };
        //! The CUDA/HIP vectors offset set trait specialization.
        template<typename TOffsets, typename TOffset>
        struct SetOffset<
            DimInt<Dim<TOffsets>::value - 3u>,
            TOffsets,
            TOffset,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TOffsets> && (Dim<TOffsets>::value >= 3)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(TOffsets const& offsets, TOffset const& offset) -> void
            {
                offsets.z = offset;
            }
        };
        //! The CUDA/HIP vectors offset set trait specialization.
        template<typename TOffsets, typename TOffset>
        struct SetOffset<
            DimInt<Dim<TOffsets>::value - 4u>,
            TOffsets,
            TOffset,
            std::enable_if_t<alpaka::detail::isCudaHipBuiltInType<TOffsets> && (Dim<TOffsets>::value >= 4)>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto setOffset(TOffsets const& offsets, TOffset const& offset) -> void
            {
                offsets.w = offset;
            }
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
