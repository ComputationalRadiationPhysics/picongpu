/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED

#    include <alpaka/core/BoostPredef.hpp>

#    if BOOST_COMP_MSVC
#        pragma warning(push)

#        pragma warning(disable : 4100) // boost/context/detail/apply.hpp(31): warning C4100: "tpl": unreferenced
                                        // formal parameter
#        pragma warning(disable : 4245) // boost/fiber/detail/futex.hpp(52): warning C4245: 'argument': conversion from
                                        // 'int' to 'DWORD', signed/unsigned mismatch
#        pragma warning(disable : 4324) // boost/fiber/detail/context_mpsc_queue.hpp(41): warning C4324:
                                        // 'boost::fibers::detail::context_mpsc_queue': structure was padded due to
                                        // alignment specifier
#        pragma warning(disable : 4456) // boost/context/execution_context_v2.hpp(301): warning C4456: declaration of
                                        // 'p' hides previous local declaration
#        pragma warning(disable : 4702) // boost/context/execution_context_v2.hpp(49): warning C4702: unreachable code
// Boost.Fiber indirectly includes windows.h for which we need to define some things.
#        define NOMINMAX
#    endif

// Boost fiber:
// http://www.boost.org/doc/libs/develop/libs/fiber/doc/html/index.html
// https://github.com/boostorg/fiber
#    include <boost/fiber/barrier.hpp>
#    include <boost/fiber/condition_variable.hpp>
#    include <boost/fiber/fiber.hpp>
#    include <boost/fiber/future.hpp>
#    include <boost/fiber/mutex.hpp>
#    include <boost/fiber/operations.hpp>

#    if BOOST_COMP_MSVC
#        undef NOMINMAX
#        pragma warning(pop)
#    endif

#endif
