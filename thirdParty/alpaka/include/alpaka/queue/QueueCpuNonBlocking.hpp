/* Copyright 2020 Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/event/EventCpu.hpp"
#include "alpaka/queue/QueueGenericThreadsNonBlocking.hpp"

namespace alpaka
{
    using QueueCpuNonBlocking = QueueGenericThreadsNonBlocking<DevCpu>;
} // namespace alpaka
