/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/ChargeDensity.def"
#include "picongpu/particles/particleToGrid/derivedAttributes/IsWeighted.hpp"
#include "picongpu/traits/attribute/GetCharge.hpp"

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
                DINLINE float_X ChargeDensity::operator()(T_Particle& particle) const
                {
                    /* read existing attributes */
                    float_X const weighting = particle[weighting_];
                    float_X const charge = picongpu::traits::attribute::getCharge(weighting, particle);

                    /* calculate new attribute */
                    float_X const particleChargeDensity = charge / sim.pic.getCellSize().productOfComponents();

                    /* return attribute */
                    return particleChargeDensity;
                }

                //! Charge density is weighted
                template<>
                struct IsWeighted<ChargeDensity> : std::true_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
