/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/queue/cpu/IGenericThreadsQueue.hpp"

namespace alpaka::cpu
{
    //! The CPU queue interface
    using ICpuQueue = IGenericThreadsQueue<DevCpu>;
} // namespace alpaka::cpu
