/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <cstdint>

namespace pmacc
{
    namespace eventSystem
    {
        /**
         * Internal event/task type used for notifications in the event system.
         */
        enum EventType
        {
            FINISHED,
            COPY,
            SENDFINISHED,
            RECVFINISHED,
            LOGICALAND,
            SETVALUE,
            GETVALUE,
            KERNEL,
            SIGNAL
        };

    } // namespace eventSystem

    // for backward compatibility pull all definitions into the pmacc namespace
    using namespace eventSystem;
} // namespace pmacc
