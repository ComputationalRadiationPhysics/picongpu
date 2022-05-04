/* Copyright 2020 Jeffrey Kelling, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/event/EventCpu.hpp>
#include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>

namespace alpaka
{
    using QueueCpuNonBlocking = QueueGenericThreadsNonBlocking<DevCpu>;
}
