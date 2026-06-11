/*
 * SPDX-FileCopyrightText: Jakob Trojok
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** calculates ionization current
             */
            struct JIonizationCalc
            {
                /** Functor calculating ionization current.
                 * Is only called if ionization energy is not zero,
                 * thus we ensure the field is different from zero.
                 */
                HDINLINE float3_X operator()(float_X const ionizationEnergy, float3_X const eField)
                {
                    float3_X jion = ionizationEnergy * eField / pmacc::math::l2norm2(eField) / sim.pic.getDt()
                                    / sim.pic.getCellSize().productOfComponents();
                    return jion;
                }
            };
        } // namespace ionization
    } // namespace particles
} // namespace picongpu
