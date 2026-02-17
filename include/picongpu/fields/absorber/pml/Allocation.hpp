/* Copyright 2026 PIConGPU contributors
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/absorber/Thickness.hpp"

#include <pmacc/assert.hpp>

namespace picongpu::fields::absorber::pml
{
    /** Compute PML thickness used for psi allocation on this rank.
     *
     * The resulting thickness is guaranteed to fit the local domain:
     * allocationNegative + allocationPositive <= localGridSize for each axis.
     */
    HINLINE Thickness
    computeAllocationThickness(GridLayout<simDim> const& gridLayout, Thickness const& globalThickness)
    {
        Thickness result = globalThickness;
        auto const localGridSize = gridLayout.sizeWithoutGuardND();
        for(uint32_t dim = 0u; dim < simDim; ++dim)
        {
            auto const localSize = localGridSize[dim];
            auto negative = result(dim, 0);
            auto positive = result(dim, 1);

            if(negative > localSize)
                negative = localSize;
            if(positive > localSize)
                positive = localSize;

            auto const total = negative + positive;
            if(total > localSize)
            {
                PMACC_ASSERT_MSG(
                    false,
                    "PML allocation thickness exceeds local grid size on this rank; clamping psi allocation.");
                auto const overflow = total - localSize;
                auto const positiveReduction = (overflow < positive) ? overflow : positive;
                positive -= positiveReduction;
                negative -= overflow - positiveReduction;
            }

            result(dim, 0) = negative;
            result(dim, 1) = positive;
        }
        return result;
    }
} // namespace picongpu::fields::absorber::pml
