/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
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


#define ALPAKA_ASSERT(EXPRESSION) assert(EXPRESSION)

namespace alpaka
{
    namespace core
    {
        namespace detail
        {
            //#############################################################################
            template<typename TArg, typename TSfinae = void>
            struct AssertValueUnsigned;
            //#############################################################################
            template<typename TArg>
            struct AssertValueUnsigned<TArg, std::enable_if_t<!std::is_unsigned<TArg>::value>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto assertValueUnsigned(TArg const& arg) -> void
                {
#ifdef NDEBUG
                    alpaka::ignore_unused(arg);
#else
                    ALPAKA_ASSERT(arg >= 0);
#endif
                }
            };
            //#############################################################################
            template<typename TArg>
            struct AssertValueUnsigned<TArg, std::enable_if_t<std::is_unsigned<TArg>::value>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto assertValueUnsigned(TArg const& arg) -> void
                {
                    alpaka::ignore_unused(arg);
                    // Nothing to do for unsigned types.
                }
            };
        } // namespace detail
        //-----------------------------------------------------------------------------
        //! This method checks integral values if they are greater or equal zero.
        //! The implementation prevents warnings for checking this for unsigned types.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TArg>
        ALPAKA_FN_HOST_ACC auto assertValueUnsigned(TArg const& arg) -> void
        {
            detail::AssertValueUnsigned<TArg>::assertValueUnsigned(arg);
        }

        namespace detail
        {
            //#############################################################################
            template<typename TLhs, typename TRhs, typename TSfinae = void>
            struct AssertGreaterThan;
            //#############################################################################
            template<typename TLhs, typename TRhs>
            struct AssertGreaterThan<
                TLhs,
                TRhs,
                std::enable_if_t<!std::is_unsigned<TRhs>::value || (TLhs::value != 0u)>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto assertGreaterThan(TRhs const& lhs) -> void
                {
#ifdef NDEBUG
                    alpaka::ignore_unused(lhs);
#else
                    ALPAKA_ASSERT(TLhs::value > lhs);
#endif
                }
            };
            //#############################################################################
            template<typename TLhs, typename TRhs>
            struct AssertGreaterThan<
                TLhs,
                TRhs,
                std::enable_if_t<std::is_unsigned<TRhs>::value && (TLhs::value == 0u)>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto assertGreaterThan(TRhs const& lhs) -> void
                {
                    alpaka::ignore_unused(lhs);
                    // Nothing to do for unsigned types camparing to zero.
                }
            };
        } // namespace detail
        //-----------------------------------------------------------------------------
        //! This method asserts that the integral value TArg is less than Tidx.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TLhs, typename TRhs>
        ALPAKA_FN_HOST_ACC auto assertGreaterThan(TRhs const& lhs) -> void
        {
            detail::AssertGreaterThan<TLhs, TRhs>::assertGreaterThan(lhs);
        }
    } // namespace core
} // namespace alpaka
