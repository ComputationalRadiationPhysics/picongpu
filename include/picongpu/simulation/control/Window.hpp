/* Copyright 2013-2021 Rene Widera, Felix Schmitt
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/types.hpp>
#include <pmacc/mappings/simulation/Selection.hpp>

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
