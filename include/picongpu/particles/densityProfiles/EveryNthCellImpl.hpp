/* Copyright 2017-2021 Axel Huebl
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

#include "picongpu/particles/densityProfiles/EveryNthCellImpl.def"
#include "picongpu/simulation_defines.hpp"

#include <pmacc/math/Vector.hpp>


namespace picongpu
{
    namespace densityProfiles
    {
        template<uint32_t... Args>
        struct EveryNthCellImpl<pmacc::math::CT::UInt32<Args...>>
        {
            using OrgSkipCells = pmacc::math::CT::UInt32<Args...>;
            using SkipCells = typename pmacc::math::CT::shrinkTo<OrgSkipCells, simDim>::type;

            template<typename T_SpeciesType>
            struct apply
            {
                using type = EveryNthCellImpl<OrgSkipCells>;
            };

            HINLINE
            EveryNthCellImpl(uint32_t currentStep)
            {
            }

            /** Calculate the normalized density
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
            {
                // modulo!
                auto const isThisCellWithProbe(totalCellOffset % SkipCells::toRT());

                // is this cell populated with a probe particle?
                bool const isPopulated(isThisCellWithProbe == DataSpace<simDim>::create(0));

                /* every how many (volumentric) cells do we set a particle:
                 * scale up weighting accordingly */
                float_X const weightingScaling(precisionCast<float_X>(SkipCells::toRT().productOfComponents()));

                // fill only the selected cells
                float_X result(0.0);
                if(isPopulated)
                    result = weightingScaling;

                return result;
            }
        };
    } // namespace densityProfiles
} // namespace picongpu
