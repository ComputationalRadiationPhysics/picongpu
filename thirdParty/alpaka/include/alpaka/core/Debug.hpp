/* Copyright 2022 Alexander Matthes, Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

#include <iostream>
#include <string>
#include <utility>

//! The no debug level.
#define ALPAKA_DEBUG_DISABLED 0
//! The minimal debug level.
#define ALPAKA_DEBUG_MINIMAL 1
//! The full debug level.
#define ALPAKA_DEBUG_FULL 2

#ifndef ALPAKA_DEBUG
//! Set the minimum log level if it is not defined.
#    define ALPAKA_DEBUG ALPAKA_DEBUG_DISABLED
#endif

namespace alpaka::core::detail
{
    //! Scope logger.
    class ScopeLogStdOut final
    {
    public:
        explicit ScopeLogStdOut(std::string sScope) : m_sScope(std::move(sScope))
        {
            std::cout << "[+] " << m_sScope << std::endl;
        }

        ScopeLogStdOut(ScopeLogStdOut const&) = delete;
        ScopeLogStdOut(ScopeLogStdOut&&) = delete;
        auto operator=(ScopeLogStdOut const&) -> ScopeLogStdOut& = delete;
        auto operator=(ScopeLogStdOut&&) -> ScopeLogStdOut& = delete;

        ~ScopeLogStdOut()
        {
            std::cout << "[-] " << m_sScope << std::endl;
        }

    private:
        std::string const m_sScope;
    };
} // namespace alpaka::core::detail

// Define ALPAKA_DEBUG_MINIMAL_LOG_SCOPE.
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    define ALPAKA_DEBUG_MINIMAL_LOG_SCOPE ::alpaka::core::detail::ScopeLogStdOut const scopeLogStdOut(__func__)
#else
#    define ALPAKA_DEBUG_MINIMAL_LOG_SCOPE
#endif

// Define ALPAKA_DEBUG_FULL_LOG_SCOPE.
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
#    define ALPAKA_DEBUG_FULL_LOG_SCOPE ::alpaka::core::detail::ScopeLogStdOut const scopeLogStdOut(__func__)
#else
#    define ALPAKA_DEBUG_FULL_LOG_SCOPE
#endif

// Define ALPAKA_DEBUG_BREAK.
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    if BOOST_COMP_GNUC || BOOST_COMP_CLANG
#        define ALPAKA_DEBUG_BREAK ::__builtin_trap()
#    elif BOOST_COMP_MSVC
#        define ALPAKA_DEBUG_BREAK ::__debugbreak()
#    else
#        define ALPAKA_DEBUG_BREAK
  // #error debug-break for current compiler not implemented!
#    endif
#else
#    define ALPAKA_DEBUG_BREAK
#endif
