/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/elem/Traits.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/meta/IntegerSequence.hpp>
#include <alpaka/meta/Metafunctions.hpp>

#include <hip/hip_runtime.h>

#include <array>
#include <type_traits>
#include <utility>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstddef>

#ifdef __HIP_PLATFORM_HCC__
  #define HIPRT_CB
#endif


namespace alpaka
{
    namespace hip
    {
        namespace detail
        {
            //-----------------------------------------------------------------------------
            //! HIP runtime API error checking with log and exception, ignoring specific error values
            ALPAKA_FN_HOST inline auto hipRtCheck(
                hipError_t const & error,
                char const * desc,
                char const * file,
                int const & line)
            -> void
            {
                if(error != hipSuccess)
                {
                    std::string const sError(std::string(file) + "(" + std::to_string(line) + ") " + std::string(desc) + " : '" + hipGetErrorName(error) +  "': '" + std::string(hipGetErrorString(error)) + "'!");
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    std::cerr << sError << std::endl;
#endif
                    ALPAKA_DEBUG_BREAK;
                    throw std::runtime_error(sError);
                }
            }
            //-----------------------------------------------------------------------------
            //! HIP runtime API error checking with log and exception, ignoring specific error values
            // NOTE: All ignored errors have to be convertible to hipError_t.
            template<
                typename... TErrors/*,
                typename = typename std::enable_if<
                    meta::Conjunction<
                        std::true_type,
                        std::is_convertible<
                            TErrors,
                            hipError_t
                        >...
                    >::value>::type*/>
            ALPAKA_FN_HOST auto hipRtCheckIgnore(
                hipError_t const & error,
                char const * cmd,
                char const * file,
                int const & line,
                TErrors && ... ignoredErrorCodes)
            -> void
            {
                if(error != hipSuccess)
                {
                    // https://stackoverflow.com/questions/18792731/can-we-omit-the-double-braces-for-stdarray-in-c14/18792782#18792782
                    std::array<hipError_t, sizeof...(ignoredErrorCodes)> const aIgnoredErrorCodes{{ignoredErrorCodes...}};
                    // If the error code is not one of the ignored ones.
                    if(std::find(aIgnoredErrorCodes.cbegin(), aIgnoredErrorCodes.cend(), error) == aIgnoredErrorCodes.cend())
                    {
                        hipRtCheck(error, ("'" + std::string(cmd) + "' returned error ").c_str(), file, line);
                    }
                }
            }
            //-----------------------------------------------------------------------------
            //! HIP runtime API last error checking with log and exception.
            ALPAKA_FN_HOST inline auto hipRtCheckLastError(
                char const * desc,
                char const * file,
                int const & line)
            -> void
            {
                hipError_t const error(hipGetLastError());
                hipRtCheck(error, desc, file, line);
            }
        }
    }
}

#if BOOST_COMP_MSVC
    //-----------------------------------------------------------------------------
    //! HIP runtime error checking with log and exception, ignoring specific error values
    #define ALPAKA_HIP_RT_CHECK_IGNORE(cmd, ...)\
        ::alpaka::hip::detail::hipRtCheckLastError("'" #cmd "' A previous HIP call (not this one) set the error ", __FILE__, __LINE__);\
        ::alpaka::hip::detail::hipRtCheckIgnore(cmd, #cmd, __FILE__, __LINE__, __VA_ARGS__)
#else
    #if BOOST_COMP_CLANG
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
    #endif
    //-----------------------------------------------------------------------------
    //! HIP runtime error checking with log and exception, ignoring specific error values
    #define ALPAKA_HIP_RT_CHECK_IGNORE(cmd, ...)\
        ::alpaka::hip::detail::hipRtCheckLastError("'" #cmd "' A previous HIP call (not this one) set the error ", __FILE__, __LINE__);\
        ::alpaka::hip::detail::hipRtCheckIgnore(cmd, #cmd, __FILE__, __LINE__, ##__VA_ARGS__)
    #if BOOST_COMP_CLANG
        #pragma clang diagnostic pop
    #endif
#endif

//-----------------------------------------------------------------------------
//! HIP runtime error checking with log and exception.
#define ALPAKA_HIP_RT_CHECK(cmd)\
    ALPAKA_HIP_RT_CHECK_IGNORE(cmd)

//-----------------------------------------------------------------------------
// HIP vector_types.h trait specializations.
namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The HIP specifics.
    namespace hip
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP vectors 1D dimension get trait specialization.
            template<
                typename T>
            struct IsHipBuiltInType :
                std::integral_constant<
                    bool,
                    std::is_same<T, char1>::value
                    || std::is_same<T, double1>::value
                    || std::is_same<T, float1>::value
                    || std::is_same<T, int1>::value
                    || std::is_same<T, long1>::value
                    || std::is_same<T, longlong1>::value
                    || std::is_same<T, short1>::value
                    || std::is_same<T, uchar1>::value
                    || std::is_same<T, uint1>::value
                    || std::is_same<T, ulong1>::value
                    || std::is_same<T, ulonglong1>::value
                    || std::is_same<T, ushort1>::value
                    || std::is_same<T, char2>::value
                    || std::is_same<T, double2>::value
                    || std::is_same<T, float2>::value
                    || std::is_same<T, int2>::value
                    || std::is_same<T, long2>::value
                    || std::is_same<T, longlong2>::value
                    || std::is_same<T, short2>::value
                    || std::is_same<T, uchar2>::value
                    || std::is_same<T, uint2>::value
                    || std::is_same<T, ulong2>::value
                    || std::is_same<T, ulonglong2>::value
                    || std::is_same<T, ushort2>::value
                    || std::is_same<T, char3>::value
                    || std::is_same<T, dim3>::value
                    || std::is_same<T, double3>::value
                    || std::is_same<T, float3>::value
                    || std::is_same<T, int3>::value
                    || std::is_same<T, long3>::value
                    || std::is_same<T, longlong3>::value
                    || std::is_same<T, short3>::value
                    || std::is_same<T, uchar3>::value
                    || std::is_same<T, uint3>::value
                    || std::is_same<T, ulong3>::value
                    || std::is_same<T, ulonglong3>::value
                    || std::is_same<T, ushort3>::value
                    || std::is_same<T, char4>::value
                    || std::is_same<T, double4>::value
                    || std::is_same<T, float4>::value
                    || std::is_same<T, int4>::value
                    || std::is_same<T, long4>::value
                    || std::is_same<T, longlong4>::value
                    || std::is_same<T, short4>::value
                    || std::is_same<T, uchar4>::value
                    || std::is_same<T, uint4>::value
                    || std::is_same<T, ulong4>::value
                    || std::is_same<T, ulonglong4>::value
                    || std::is_same<T, ushort4>::value
                >
            {};
        }
    }
    namespace dim
    {
        namespace traits
        {
          // If you receive '"alpaka::dim::traits::DimType" has already been defined'
          // then too many operators in the enable_if are used. Split them in two or more structs.
          // (compiler: gcc 5.3.0)
            //#############################################################################
            //! The HIP vectors 1D dimension get trait specialization.
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, char1>::value
                    || std::is_same<T, double1>::value
                    || std::is_same<T, float1>::value
                    || std::is_same<T, int1>::value
                    || std::is_same<T, long1>::value
                    || std::is_same<T, longlong1>::value
                    || std::is_same<T, short1>::value
                >::type>
            {
                using type = dim::DimInt<1u>;
            };
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, uchar1>::value
                    || std::is_same<T, uint1>::value
                    || std::is_same<T, ulong1>::value
                    || std::is_same<T, ulonglong1>::value
                    || std::is_same<T, ushort1>::value
                >::type>
            {
                using type = dim::DimInt<1u>;
            };
            //#############################################################################
            //! The HIP vectors 2D dimension get trait specialization.
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, char2>::value
                    || std::is_same<T, double2>::value
                    || std::is_same<T, float2>::value
                    || std::is_same<T, int2>::value
                    || std::is_same<T, long2>::value
                    || std::is_same<T, longlong2>::value
                    || std::is_same<T, short2>::value
                >::type>
            {
                using type = dim::DimInt<2u>;
            };
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, uchar2>::value
                    || std::is_same<T, uint2>::value
                    || std::is_same<T, ulong2>::value
                    || std::is_same<T, ulonglong2>::value
                    || std::is_same<T, ushort2>::value
                >::type>
            {
                using type = dim::DimInt<2u>;
            };
            //#############################################################################
            //! The HIP vectors 3D dimension get trait specialization.
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, char3>::value
                    || std::is_same<T, dim3>::value
                    || std::is_same<T, double3>::value
                    || std::is_same<T, float3>::value
                    || std::is_same<T, int3>::value
                    || std::is_same<T, long3>::value
                    || std::is_same<T, longlong3>::value
                    || std::is_same<T, short3>::value
                >::type>
            {
                using type = dim::DimInt<3u>;
            };
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, uchar3>::value
                    || std::is_same<T, uint3>::value
                    || std::is_same<T, ulong3>::value
                    || std::is_same<T, ulonglong3>::value
                    || std::is_same<T, ushort3>::value
                >::type>
            {
                using type = dim::DimInt<3u>;
            };
            //#############################################################################
            //! The HIP vectors 4D dimension get trait specialization.
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, char4>::value
                    || std::is_same<T, double4>::value
                    || std::is_same<T, float4>::value
                    || std::is_same<T, int4>::value
                    || std::is_same<T, long4>::value
                    || std::is_same<T, longlong4>::value
                    || std::is_same<T, short4>::value
                >::type>
            {
                using type = dim::DimInt<4u>;
            };
            template<
                typename T>
            struct DimType<
                T,
                typename std::enable_if<
                    std::is_same<T, uchar4>::value
                    || std::is_same<T, uint4>::value
                    || std::is_same<T, ulong4>::value
                    || std::is_same<T, ulonglong4>::value
                    || std::is_same<T, ushort4>::value
                >::type>
            {
                using type = dim::DimInt<4u>;
            };
        }
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP vectors elem type trait specialization.
            template<
                typename T>
            struct ElemType<
                T,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<T>::value>::type>
            {
                using type = decltype(std::declval<T>().x);
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP vectors extent get trait specialization.
            template<
                typename TExtent>
            struct GetExtent<
                dim::DimInt<dim::Dim<TExtent>::value - 1u>,
                TExtent,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TExtent>::value
                    && (dim::Dim<TExtent>::value >= 1)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtent const & extent)
                -> decltype(extent.x)
                {
                    return extent.x;
                }
            };
            //#############################################################################
            //! The HIP vectors extent get trait specialization.
            template<
                typename TExtent>
            struct GetExtent<
                dim::DimInt<dim::Dim<TExtent>::value - 2u>,
                TExtent,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TExtent>::value
                    && (dim::Dim<TExtent>::value >= 2)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtent const & extent)
                -> decltype(extent.y)
                {
                    return extent.y;
                }
            };
            //#############################################################################
            //! The HIP vectors extent get trait specialization.
            template<
                typename TExtent>
            struct GetExtent<
                dim::DimInt<dim::Dim<TExtent>::value - 3u>,
                TExtent,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TExtent>::value
                    && (dim::Dim<TExtent>::value >= 3)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtent const & extent)
                -> decltype(extent.z)
                {
                    return extent.z;
                }
            };
            //#############################################################################
            //! The HIP vectors extent get trait specialization.
            template<
                typename TExtent>
            struct GetExtent<
                dim::DimInt<dim::Dim<TExtent>::value - 4u>,
                TExtent,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TExtent>::value
                    && (dim::Dim<TExtent>::value >= 4)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtent const & extent)
                -> decltype(extent.w)
                {
                    return extent.w;
                }
            };
            //#############################################################################
            //! The HIP vectors extent set trait specialization.
            template<
                typename TExtent,
                typename TExtentVal>
            struct SetExtent<
                dim::DimInt<dim::Dim<TExtent>::value - 1u>,
                TExtent,
                TExtentVal,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TExtent>::value
                    && (dim::Dim<TExtent>::value >= 1)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    TExtent const & extent,
                    TExtentVal const & extentVal)
                -> void
                {
                    extent.x = extentVal;
                }
            };
            //#############################################################################
            //! The HIP vectors extent set trait specialization.
            template<
                typename TExtent,
                typename TExtentVal>
            struct SetExtent<
                dim::DimInt<dim::Dim<TExtent>::value - 2u>,
                TExtent,
                TExtentVal,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TExtent>::value
                    && (dim::Dim<TExtent>::value >= 2)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    TExtent const & extent,
                    TExtentVal const & extentVal)
                -> void
                {
                    extent.y = extentVal;
                }
            };
            //#############################################################################
            //! The HIP vectors extent set trait specialization.
            template<
                typename TExtent,
                typename TExtentVal>
            struct SetExtent<
                dim::DimInt<dim::Dim<TExtent>::value - 3u>,
                TExtent,
                TExtentVal,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TExtent>::value
                    && (dim::Dim<TExtent>::value >= 3)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    TExtent const & extent,
                    TExtentVal const & extentVal)
                -> void
                {
                    extent.z = extentVal;
                }
            };
            //#############################################################################
            //! The HIP vectors extent set trait specialization.
            template<
                typename TExtent,
                typename TExtentVal>
            struct SetExtent<
                dim::DimInt<dim::Dim<TExtent>::value - 4u>,
                TExtent,
                TExtentVal,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TExtent>::value
                    && (dim::Dim<TExtent>::value >= 4)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    TExtent const & extent,
                    TExtentVal const & extentVal)
                -> void
                {
                    extent.w = extentVal;
                }
            };
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP vectors offset get trait specialization.
            template<
                typename TOffsets>
            struct GetOffset<
                dim::DimInt<dim::Dim<TOffsets>::value - 1u>,
                TOffsets,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TOffsets>::value
                    && (dim::Dim<TOffsets>::value >= 1)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    TOffsets const & offsets)
                -> decltype(offsets.x)
                {
                    return offsets.x;
                }
            };
            //#############################################################################
            //! The HIP vectors offset get trait specialization.
            template<
                typename TOffsets>
            struct GetOffset<
                dim::DimInt<dim::Dim<TOffsets>::value - 2u>,
                TOffsets,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TOffsets>::value
                    && (dim::Dim<TOffsets>::value >= 2)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    TOffsets const & offsets)
                -> decltype(offsets.y)
                {
                    return offsets.y;
                }
            };
            //#############################################################################
            //! The HIP vectors offset get trait specialization.
            template<
                typename TOffsets>
            struct GetOffset<
                dim::DimInt<dim::Dim<TOffsets>::value - 3u>,
                TOffsets,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TOffsets>::value
                    && (dim::Dim<TOffsets>::value >= 3)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    TOffsets const & offsets)
                -> decltype(offsets.z)
                {
                    return offsets.z;
                }
            };
            //#############################################################################
            //! The HIP vectors offset get trait specialization.
            template<
                typename TOffsets>
            struct GetOffset<
                dim::DimInt<dim::Dim<TOffsets>::value - 4u>,
                TOffsets,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TOffsets>::value
                    && (dim::Dim<TOffsets>::value >= 4)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    TOffsets const & offsets)
                -> decltype(offsets.w)
                {
                    return offsets.w;
                }
            };
            //#############################################################################
            //! The HIP vectors offset set trait specialization.
            template<
                typename TOffsets,
                typename TOffset>
            struct SetOffset<
                dim::DimInt<dim::Dim<TOffsets>::value - 1u>,
                TOffsets,
                TOffset,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TOffsets>::value
                    && (dim::Dim<TOffsets>::value >= 1)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setOffset(
                    TOffsets const & offsets,
                    TOffset const & offset)
                -> void
                {
                    offsets.x = offset;
                }
            };
            //#############################################################################
            //! The HIP vectors offset set trait specialization.
            template<
                typename TOffsets,
                typename TOffset>
            struct SetOffset<
                dim::DimInt<dim::Dim<TOffsets>::value - 2u>,
                TOffsets,
                TOffset,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TOffsets>::value
                    && (dim::Dim<TOffsets>::value >= 2)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setOffset(
                    TOffsets const & offsets,
                    TOffset const & offset)
                -> void
                {
                    offsets.y = offset;
                }
            };
            //#############################################################################
            //! The HIP vectors offset set trait specialization.
            template<
                typename TOffsets,
                typename TOffset>
            struct SetOffset<
                dim::DimInt<dim::Dim<TOffsets>::value - 3u>,
                TOffsets,
                TOffset,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TOffsets>::value
                    && (dim::Dim<TOffsets>::value >= 3)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setOffset(
                    TOffsets const & offsets,
                    TOffset const & offset)
                -> void
                {
                    offsets.z = offset;
                }
            };
            //#############################################################################
            //! The HIP vectors offset set trait specialization.
            template<
                typename TOffsets,
                typename TOffset>
            struct SetOffset<
                dim::DimInt<dim::Dim<TOffsets>::value - 4u>,
                TOffsets,
                TOffset,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TOffsets>::value
                    && (dim::Dim<TOffsets>::value >= 4)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setOffset(
                    TOffsets const & offsets,
                    TOffset const & offset)
                -> void
                {
                    offsets.w = offset;
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP vectors idx type trait specialization.
            template<
                typename TIdx>
            struct IdxType<
                TIdx,
                typename std::enable_if<
                    hip::traits::IsHipBuiltInType<TIdx>::value>::type>
            {
                using type = std::size_t;
            };
        }
    }
}

#endif
