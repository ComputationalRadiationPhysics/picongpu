/*
 * SPDX-FileCopyrightText: Rene Widera, Felix Schmitt
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/mappings/simulation/Selection.hpp>
#include <pmacc/types.hpp>

namespace picongpu
{
    using namespace pmacc;

    /**
     * Window describes sizes and offsets.
     *
     * For a detailed description of windows, see the PIConGPU wiki page:
     * https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions
     */
    struct Window
    {
        /* Dimensions (size/offset) of the global virtual window over all GPUs */
        Selection<simDim> globalDimensions;

        /* Dimensions (size/offset) of the local virtual window on this GPU */
        Selection<simDim> localDimensions;
    };
} // namespace picongpu
