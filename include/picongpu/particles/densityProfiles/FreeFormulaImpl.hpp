/*
 * SPDX-FileCopyrightText: Rene Widera, Richard Pausch, Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/densityProfiles/IProfile.hpp"
#include "picongpu/particles/functor/User.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"

namespace picongpu
{
    namespace densityProfiles
    {
        template<typename T_UserFunctor>
        struct FreeFormulaImpl : public particles::functor::User<T_UserFunctor>
        {
            using UserFunctor = particles::functor::User<T_UserFunctor>;

            template<typename T_SpeciesType>
            struct apply
            {
                using type = FreeFormulaImpl<UserFunctor>;
            };

            HINLINE FreeFormulaImpl(uint32_t currentStep, IdGenerator idGen) : UserFunctor(currentStep, idGen)
            {
            }

            /** Calculate the normalized density
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
            {
                float_64 const unitLength(sim.unit.length()); // workaround to use sim.unit.length() on device
                float3_64 const cellSize_SI(precisionCast<float_64>(sim.pic.getCellSize()) * unitLength);
                // evaluate at cell center for a more accurate estimate for the cell
                floatD_64 const totalCenterCellOffset
                    = precisionCast<float_64>(totalCellOffset) + floatD_64::create(0.5);
                floatD_64 const position_SI(totalCenterCellOffset * cellSize_SI.shrink<simDim>());

                return UserFunctor::operator()(position_SI, cellSize_SI);
            }
        };
    } // namespace densityProfiles
} // namespace picongpu
