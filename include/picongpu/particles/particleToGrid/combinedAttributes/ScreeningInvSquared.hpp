/*
 * SPDX-FileCopyrightText: Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/combinedAttributes/AverageAttribute.def"
#include "picongpu/particles/particleToGrid/combinedAttributes/ScreeningInvSquared.def"

#include <limits>
#include <string>
#include <vector>

namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace combinedAttributes
            {
                /** Functor implementation
                 *
                 * Result overwrites the chargeDensity value.
                 *
                 * @tparam T_Acc alpaka accelerator type
                 * @param acc alpaka accelerator
                 * @param chargeDensity charge density value and the result destination
                 * @param energyDensity energy density value
                 */
                template<typename T_Acc>
                HDINLINE void ScreeningInvSquaredOperation::operator()(
                    T_Acc const& acc,
                    float1_X& chargeDensity,
                    float1_X const& energyDensity) const
                {
                    // charge density = 0 means either neutral particles or no particles
                    // in that case set the inverse to zero so that it is not contributing to the total Debye length
                    // calculation
                    if(math::abs(chargeDensity[0]) <= std::numeric_limits<float_X>::min())
                    {
                        chargeDensity[0] = 0;
                    }
                    // non-zero chargeDenisty but zero temperature
                    // the Debye length approaches zero so the inverse approaches infinity
                    else if(energyDensity[0] <= std::numeric_limits<float_X>::min())
                    {
                        chargeDensity[0] = std::numeric_limits<float_X>::infinity();
                    }
                    else
                        // T = 2/3 average_energy
                        chargeDensity = (3.0_X / 2.0_X) * (1.0_X / sim.pic.getEps0()) * chargeDensity * chargeDensity
                                        / energyDensity;
                }

                struct ScreeningInvSquaredDescription
                {
                    HDINLINE float1_64 getUnit() const
                    {
                        // inverse squared screening length has unit:
                        return 1.0 / (sim.unit.length() * sim.unit.length());
                    }

                    HINLINE std::vector<float_64> getUnitDimension() const
                    {
                        /* L, M, T, I, theta, N, J
                         *
                         *  inverse squared meter: m^-2
                         *   -> L^-2
                         */
                        std::vector<float_64> unitDimension(7, 0.0);
                        unitDimension.at(SIBaseUnits::length) = -2.0;

                        return unitDimension;
                    }

                    HINLINE static std::string getName()
                    {
                        return "invSquaredScreenLength";
                    }
                };

            } // namespace combinedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
