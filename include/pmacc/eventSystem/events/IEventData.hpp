/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/eventSystem/events/EventNotify.hpp"

namespace pmacc
{
    // forward declaration
    class EventNotify;

    /**
     * Base class for event data.
     */
    class IEventData
    {
    public:
        IEventData(EventNotify* task) : task(task)
        {
        }

        virtual ~IEventData() = default;

        EventNotify* getEventNotify()
        {
            return task;
        }

    protected:
        EventNotify* task;
    };

} // namespace pmacc
