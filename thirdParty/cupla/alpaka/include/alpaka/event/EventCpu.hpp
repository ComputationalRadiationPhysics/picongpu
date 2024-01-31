/* Copyright 2020 Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/event/EventGenericThreads.hpp"

namespace alpaka
{
    using EventCpu = EventGenericThreads<DevCpu>;
} // namespace alpaka
