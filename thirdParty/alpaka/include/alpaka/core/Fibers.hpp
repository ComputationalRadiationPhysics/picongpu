/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED

#include <alpaka/core/Common.hpp>

#if BOOST_COMP_MSVC
    #pragma warning(push)

    #pragma warning(disable: 4100)  // boost/context/detail/apply.hpp(31): warning C4100: "tpl": unreferenced formal parameter
    #pragma warning(disable: 4245)  // boost/fiber/detail/futex.hpp(52): warning C4245: 'argument': conversion from 'int' to 'DWORD', signed/unsigned mismatch
    #pragma warning(disable: 4324)  // boost/fiber/detail/context_mpsc_queue.hpp(41): warning C4324: 'boost::fibers::detail::context_mpsc_queue': structure was padded due to alignment specifier
    #pragma warning(disable: 4456)  // boost/context/execution_context_v2.hpp(301): warning C4456: declaration of 'p' hides previous local declaration
    #pragma warning(disable: 4702)  // boost/context/execution_context_v2.hpp(49): warning C4702: unreachable code
    // Boost.Fiber indirectly includes windows.h for which we need to define some things.
    #define NOMINMAX
#endif

// Boost fiber:
// http://www.boost.org/doc/libs/develop/libs/fiber/doc/html/index.html
// https://github.com/boostorg/fiber
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/operations.hpp>
#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/mutex.hpp>
#include <boost/fiber/future.hpp>
#include <boost/fiber/barrier.hpp>

#if BOOST_COMP_MSVC
    #undef NOMINMAX
    #pragma warning(pop)
#endif

#endif
