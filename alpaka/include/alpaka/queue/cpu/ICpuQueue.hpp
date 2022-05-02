/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>

namespace alpaka::cpu
{
    //! The CPU queue interface
    using ICpuQueue = IGenericThreadsQueue<DevCpu>;
} // namespace alpaka::cpu
