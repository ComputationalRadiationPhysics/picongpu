/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/densityProfiles/EveryNthCellImpl.def"

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
            EveryNthCellImpl(uint32_t)
            {
            }

            /** Calculate the normalized density
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
            {
                // modulo!
                auto const isThisCellWithProbe(totalCellOffset % precisionCast<int>(SkipCells::toRT()));

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
