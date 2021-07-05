/* Copyright 2015-2021 Alexander Grund, Pawel Ordyna
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/plugins/photonDetector/DetectorParams.def"

#include <pmacc/math/Vector.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace photonDetector
        {
            DetectorParams::DetectorParams(
                const DataSpace<DIM2>& size,
                const pmacc::math::Vector<float_X, DIM2>& anglePerCell,
                const DetectorPlacement& detectorPlacement)
                : size(size)
                , anglePerCell(anglePerCell)
                , detectorPlacement(detectorPlacement)
            {
            }
        } // namespace photonDetector
    } // namespace plugins
} // namespace picongpu
