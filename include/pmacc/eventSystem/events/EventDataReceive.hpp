/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/eventSystem/events/IEventData.hpp"

namespace pmacc
{
    class EventDataReceive : public IEventData
    {
    public:
        EventDataReceive(EventNotify* task, size_t recv_count) : IEventData(task), recv_count(recv_count)
        {
        }

        size_t getReceivedCount() const
        {
            return recv_count;
        }

    private:
        size_t recv_count;
    };

} // namespace pmacc
