/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt, Marco Garten, Alexander Grund, Sergei
 * Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    using namespace pmacc;

#if (!BOOST_LANG_CUDA && !BOOST_COMP_HIP)
    /* dummy because we are not using mallocMC with CPU backends
     * DeviceHeap is defined in `mallocMC.param`
     */
    struct DeviceHeap
    {
        using AllocatorHandle = int;

        int getAllocatorHandle()
        {
            return 0;
        }
    };
#endif
} // namespace picongpu
