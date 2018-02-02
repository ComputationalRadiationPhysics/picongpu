/**
 * \file
 * Copyright 2014-2017 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <boost/current_function.hpp>
#include <alpaka/core/BoostPredefWorkaround.hpp>

#include <string>
#include <iostream>

//-----------------------------------------------------------------------------
//! The no debug level.
#define ALPAKA_DEBUG_DISABLED 0
//-----------------------------------------------------------------------------
//! The minimal debug level.
#define ALPAKA_DEBUG_MINIMAL 1
//-----------------------------------------------------------------------------
//! The full debug level.
#define ALPAKA_DEBUG_FULL 2

#ifndef ALPAKA_DEBUG
    //-----------------------------------------------------------------------------
    //! Set the minimum log level if it is not defined.
    #define ALPAKA_DEBUG ALPAKA_DEBUG_DISABLED
#endif

namespace alpaka
{
    namespace core
    {
        namespace detail
        {
            //#############################################################################
            //! Scope logger.
            class ScopeLogStdOut final
            {
            public:
                //-----------------------------------------------------------------------------
                ScopeLogStdOut(
                    std::string const & sScope) :
                        m_sScope(sScope)
                {
                    std::cout << "[+] " << m_sScope << std::endl;
                }
                //-----------------------------------------------------------------------------
                ScopeLogStdOut(ScopeLogStdOut const &) = delete;
                //-----------------------------------------------------------------------------
                ScopeLogStdOut(ScopeLogStdOut &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(ScopeLogStdOut const &) -> ScopeLogStdOut & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(ScopeLogStdOut &&) -> ScopeLogStdOut & = delete;
                //-----------------------------------------------------------------------------
                ~ScopeLogStdOut()
                {
                    std::cout << "[-] " << m_sScope << std::endl;
                }

            private:
                std::string const m_sScope;
            };
        }
    }
}

//-----------------------------------------------------------------------------
// Define ALPAKA_DEBUG_MINIMAL_LOG_SCOPE.
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #define ALPAKA_DEBUG_MINIMAL_LOG_SCOPE\
        ::alpaka::core::detail::ScopeLogStdOut const scopeLogStdOut(BOOST_CURRENT_FUNCTION)
#else
    #define ALPAKA_DEBUG_MINIMAL_LOG_SCOPE
#endif

//-----------------------------------------------------------------------------
// Define ALPAKA_DEBUG_FULL_LOG_SCOPE.
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    #define ALPAKA_DEBUG_FULL_LOG_SCOPE\
        ::alpaka::core::detail::ScopeLogStdOut const scopeLogStdOut(BOOST_CURRENT_FUNCTION)
#else
    #define ALPAKA_DEBUG_FULL_LOG_SCOPE
#endif

//-----------------------------------------------------------------------------
// Define ALPAKA_DEBUG_BREAK.
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #if BOOST_COMP_GNUC
        #define ALPAKA_DEBUG_BREAK ::__builtin_trap()
    #elif BOOST_COMP_INTEL
        #define ALPAKA_DEBUG_BREAK ::__debugbreak()
    #elif BOOST_COMP_MSVC
        #define ALPAKA_DEBUG_BREAK ::__debugbreak()
    #else
        #define ALPAKA_DEBUG_BREAK
        //#error debug-break for current compiler not implemented!
    #endif
#else
    #define ALPAKA_DEBUG_BREAK
#endif
