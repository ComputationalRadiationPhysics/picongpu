/* Copyright 2022 Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>

#include <thread>
#include <utility>

namespace alpaka
{
    //! The host thread traits.
    namespace trait
    {
        //! The queue enqueue trait.
        template<typename TThread, typename TSfinae = void>
        struct IsThisThread;
    } // namespace trait

    //! Checks if the given thread handle is a handle for the executing thread.
    template<typename TThread>
    ALPAKA_FN_HOST auto isThisThread(TThread const& thread) -> bool
    {
        return trait::IsThisThread<TThread>::isThisThread(thread);
    }

    namespace trait
    {
        //! C++ STL threads implementation of IsThisThread trait.
        template<>
        struct IsThisThread<std::thread>
        {
            ALPAKA_FN_HOST static auto isThisThread(std::thread const& thread) -> bool
            {
                return std::this_thread::get_id() == thread.get_id();
            }
        };
    } // namespace trait
} // namespace alpaka
