/* Copyright 2020 Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/event/EventCpu.hpp"
#include "alpaka/queue/QueueGenericThreadsBlocking.hpp"

namespace alpaka
{
    using QueueCpuBlocking = QueueGenericThreadsBlocking<DevCpu>;
} // namespace alpaka
