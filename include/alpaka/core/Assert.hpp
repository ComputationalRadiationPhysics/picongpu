/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

#include <cassert>
#include <type_traits>

//! The assert can be explicit disabled by defining NDEBUG
#define ALPAKA_ASSERT(...) assert(__VA_ARGS__)

//! Macro which expands to a noop.
//! Macro enforces an semicolon after the call.
#define ALPAKA_NOOP(...)                                                                                              \
    do                                                                                                                \
    {                                                                                                                 \
    } while(false)

//! ALPAKA_ASSERT_ACC_IMPL is an assert-like macro.
//! It can be disabled setting the ALPAKA_DISABLE_ASSERT_ACC preprocessor symbol or the NDEBUG preprocessor symbol.
#if !defined(ALPAKA_DISABLE_ASSERT_ACC)
#    define ALPAKA_ASSERT_ACC_IMPL(...) ALPAKA_ASSERT(__VA_ARGS__)
#else
#    define ALPAKA_ASSERT_ACC_IMPL(...) ALPAKA_NOOP(__VA_ARGS__)
#endif

//! ALPAKA_ASSERT_ACC is an assert-like macro.
//!
//! In device code for a GPU or SYCL backend it can be disabled setting the ALPAKA_DISABLE_ASSERT_ACC preprocessor
//! symbol or the NDEBUG preprocessor symbol. In device code for a native C++ CPU backend and in host code, it is
//! equivalent to ALPAKA_ASSERT, and can be disabled setting the NDEBUG preprocessor symbol.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)
// CUDA device code
#    define ALPAKA_ASSERT_ACC(...) ALPAKA_ASSERT_ACC_IMPL(__VA_ARGS__)
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__)
// HIP/ROCm device code
#    define ALPAKA_ASSERT_ACC(...) ALPAKA_ASSERT_ACC_IMPL(__VA_ARGS__)
#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(__SYCL_DEVICE_ONLY__)
// SYCL/oneAPI device code
#    if defined(SYCL_EXT_ONEAPI_ASSERT)
#        define ALPAKA_ASSERT_ACC(...) ALPAKA_ASSERT_ACC_IMPL(__VA_ARGS__)
#    else
#        define ALPAKA_ASSERT_ACC(...) ALPAKA_NOOP(__VA_ARGS__)
#    endif
// add here any other #elif conditions for non-CPU backends
// ...
#else
// CPU backend, or host code
#    define ALPAKA_ASSERT_ACC(...) ALPAKA_ASSERT(__VA_ARGS__)
#endif

namespace alpaka::core
{
    namespace detail
    {
        template<typename TArg>
        struct AssertValueUnsigned
        {
            ALPAKA_NO_HOST_ACC_WARNING ALPAKA_FN_HOST_ACC static constexpr auto assertValueUnsigned(
                [[maybe_unused]] TArg const& arg)
            {
                if constexpr(std::is_signed_v<TArg>)
                    ALPAKA_ASSERT_ACC(arg >= 0);

                // Nothing to do for unsigned types.
            }
        };
    } // namespace detail

    //! This method checks integral values if they are greater or equal zero.
    //! The implementation prevents warnings for checking this for unsigned types.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC constexpr auto assertValueUnsigned(TArg const& arg) -> void
    {
        detail::AssertValueUnsigned<TArg>::assertValueUnsigned(arg);
    }

    namespace detail
    {
        template<typename TLhs, typename TRhs>
        struct AssertGreaterThan
        {
            ALPAKA_NO_HOST_ACC_WARNING ALPAKA_FN_HOST_ACC static constexpr auto assertGreaterThan(
                [[maybe_unused]] TRhs const& rhs)
            {
                if constexpr(std::is_signed_v<TRhs> || (TLhs::value != 0u))
                    ALPAKA_ASSERT_ACC(TLhs::value > rhs);

                // Nothing to do for unsigned types comparing to zero.
            }
        };
    } // namespace detail

    //! This function asserts that the integral value TLhs is greater than TRhs.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TLhs, typename TRhs>
    ALPAKA_FN_HOST_ACC constexpr auto assertGreaterThan(TRhs const& rhs) -> void
    {
        detail::AssertGreaterThan<TLhs, TRhs>::assertGreaterThan(rhs);
    }
} // namespace alpaka::core
