/* Copyright 2021-2023 Pawel Ordyna, Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/particleToGrid/combinedAttributes/AverageAttribute.def"
#include "picongpu/particles/particleToGrid/derivedAttributes/IsWeighted.hpp"

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
                template<typename T_Worker>
                HDINLINE void AverageDivideOperation::operator()(
                    T_Worker const& worker,
                    float1_X& dst,
                    const float1_X& dens) const
                {
                    // avoid dividing by zero. Return zero if density is close to zero.
                    if(dens[0] * static_cast<float_X>(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE) * CELL_VOLUME
                       <= std::numeric_limits<float_X>::min())
                    {
                        dst = float1_X{0.0};
                    }
                    else
                    {
                        // average value is total value over number of particles
                        // number of particles is density * CELL_VOLUME
                        dst /= dens * static_cast<float_X>(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE)
                            * CELL_VOLUME;
                    }
                }


                template<typename T_DerivedAttribute>
                struct AverageAttributeDescription
                {
                    // Check prerequisite on the input type
                    PMACC_CASSERT_MSG(
                        _error_average_attribute_only_supports_weighted_derived_attributes_check_trait_IsWeighted,
                        derivedAttributes::IsWeighted<T_DerivedAttribute>::value);

                    HDINLINE float1_64 getUnit() const
                    {
                        // Average quantity has the same unit as the total quantity
                        return T_DerivedAttribute().getUnit();
                    }

                    HINLINE std::vector<float_64> getUnitDimension() const
                    {
                        return T_DerivedAttribute().getUnitDimension();
                    }

                    HINLINE static std::string getName()
                    {
                        return "Average_" + T_DerivedAttribute().getName();
                    }
                };

            } // namespace combinedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
