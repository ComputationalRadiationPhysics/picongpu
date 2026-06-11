/*
 * SPDX-FileCopyrightText: Pawel Ordyna, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
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
                    float1_X const& dens) const
                {
                    // avoid dividing by zero. Return zero if density is close to zero.
                    if(dens[0] * static_cast<float_X>(sim.unit.typicalNumParticlesPerMacroParticle())
                           * sim.pic.getCellSize().productOfComponents()
                       <= std::numeric_limits<float_X>::min())
                    {
                        dst = float1_X{0.0};
                    }
                    else
                    {
                        // average value is total value over number of particles
                        // number of particles is density * sim.pic.getCellSize().productOfComponents()
                        dst /= dens * static_cast<float_X>(sim.unit.typicalNumParticlesPerMacroParticle())
                               * sim.pic.getCellSize().productOfComponents();
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
