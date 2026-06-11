/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/algorithms/Gamma.hpp"
#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/IsWeighted.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/MidCurrentDensityComponent.def"
#include "picongpu/traits/attribute/GetCharge.hpp"
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
                template<size_t T_direction>
                template<class T_Particle>
                DINLINE float_X MidCurrentDensityComponent<T_direction>::operator()(T_Particle& particle) const
                {
                    /* read existing attributes */
                    float_X const weighting = particle[weighting_];
                    float_X const charge = picongpu::traits::attribute::getCharge(weighting, particle);
                    float3_X const mom = particle[momentum_];
                    float_X const momCom = mom[T_direction];
                    float_X const mass = picongpu::traits::attribute::getMass(weighting, particle);

                    /* calculate new attribute */
                    Gamma<float_X> calcGamma;
                    typename Gamma<float_X>::valueType const gamma = calcGamma(mom, mass);

                    /* calculate new attribute */
                    float_X const particleCurrentDensity = charge / sim.pic.getCellSize().productOfComponents()
                                                           * /* rho */
                                                           momCom / (gamma * mass); /* v_component */

                    /* return attribute */
                    return particleCurrentDensity;
                }

                /** Mid current density component is weighted
                 *
                 * @param T_direction perpendicular direction x=0, y=1, z=2
                 */
                template<size_t T_direction>
                struct IsWeighted<MidCurrentDensityComponent<T_direction>> : std::true_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
