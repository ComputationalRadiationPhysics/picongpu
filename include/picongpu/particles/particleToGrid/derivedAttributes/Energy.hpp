/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/algorithms/KinEnergy.hpp"
#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/Energy.def"
#include "picongpu/particles/particleToGrid/derivedAttributes/IsWeighted.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

#include <type_traits>

namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace derivedAttributes
            {
                template<class T_Particle>
                DINLINE float_X Energy::operator()(T_Particle& particle) const
                {
                    /* read existing attributes */
                    float_X const weighting = particle[weighting_];
                    float3_X const mom = particle[momentum_];
                    float_X const mass = picongpu::traits::attribute::getMass(weighting, particle);

                    return KinEnergy<>()(mom, mass);
                }

                //! Energy is weighted
                template<>
                struct IsWeighted<Energy> : std::true_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
