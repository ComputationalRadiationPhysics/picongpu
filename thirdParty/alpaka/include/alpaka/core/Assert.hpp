/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#include <cassert>
#include <type_traits>


#if !(defined(BOOST_LANG_HIP) && BOOST_LANG_HIP && BOOST_COMP_HCC)
  #define ALPAKA_ASSERT(EXPRESSION) assert(EXPRESSION)
#else

  // Including assert.h would interfere with HIP's host-device implementation
  // see: https://github.com/ROCm-Developer-Tools/HIP/issues/599
  // However, cassert is still in some header, so we have to do a workaround for HIP.
  #ifdef NDEBUG
    #define ALPAKA_ASSERT(EXPRESSION) static_cast<void>(0)
  #else
    #define ALPAKA_ASSERT(EXPRESSION) assert_workaround(EXPRESSION)

    #pragma push_macro("__DEVICE__")
    #define __DEVICE__ extern "C" __device__ __attribute__((always_inline)) \
            __attribute__((weak))

     __DEVICE__ void __device_trap() __asm("llvm.trap");

     __host__ __device__
     __attribute__((always_inline))             \
     __attribute__((weak))
     void assert_workaround(bool expr) {
       if(!expr) {
         printf("assert failed.\n");
         #if __HIP_DEVICE_COMPILE__==1
           __device_trap();
         #else
           exit(1);
         #endif
       }
     }
  #endif //NDEBUG
#endif

namespace alpaka
{
    namespace core
    {
        namespace detail
        {
            //#############################################################################
            template<
                typename TArg,
                typename TSfinae = void>
            struct AssertValueUnsigned;
            //#############################################################################
            template<
                typename TArg>
            struct AssertValueUnsigned<
                TArg,
                typename std::enable_if<!std::is_unsigned<TArg>::value>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto assertValueUnsigned(
                    TArg const & arg)
                -> void
                {
#ifdef NDEBUG
                    alpaka::ignore_unused(arg);
#else
                    ALPAKA_ASSERT(arg >= 0);
#endif
                }
            };
            //#############################################################################
            template<
                typename TArg>
            struct AssertValueUnsigned<
                TArg,
                typename std::enable_if<std::is_unsigned<TArg>::value>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto assertValueUnsigned(
                    TArg const & arg)
                -> void
                {
                    alpaka::ignore_unused(arg);
                    // Nothing to do for unsigned types.
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! This method checks integral values if they are greater or equal zero.
        //! The implementation prevents warnings for checking this for unsigned types.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TArg>
        ALPAKA_FN_HOST_ACC auto assertValueUnsigned(
            TArg const & arg)
        -> void
        {
            detail::AssertValueUnsigned<
                TArg>
            ::assertValueUnsigned(
                arg);
        }

        namespace detail
        {
            //#############################################################################
            template<
                typename TLhs,
                typename TRhs,
                typename TSfinae = void>
            struct AssertGreaterThan;
            //#############################################################################
            template<
                typename TLhs,
                typename TRhs>
            struct AssertGreaterThan<
                TLhs,
                TRhs,
                typename std::enable_if<!std::is_unsigned<TRhs>::value || (TLhs::value != 0u)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto assertGreaterThan(
                    TRhs const & lhs)
                -> void
                {
#ifdef NDEBUG
                    alpaka::ignore_unused(lhs);
#else
                    ALPAKA_ASSERT(TLhs::value > lhs);
#endif
                }
            };
            //#############################################################################
            template<
                typename TLhs,
                typename TRhs>
            struct AssertGreaterThan<
                TLhs,
                TRhs,
                typename std::enable_if<std::is_unsigned<TRhs>::value && (TLhs::value == 0u)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto assertGreaterThan(
                    TRhs const & lhs)
                -> void
                {
                    alpaka::ignore_unused(lhs);
                    // Nothing to do for unsigned types camparing to zero.
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! This method asserts that the integral value TArg is less than Tidx.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TLhs,
            typename TRhs>
        ALPAKA_FN_HOST_ACC auto assertGreaterThan(
            TRhs const & lhs)
        -> void
        {
            detail::AssertGreaterThan<
                TLhs,
                TRhs>
            ::assertGreaterThan(
                lhs);
        }
    }
}
