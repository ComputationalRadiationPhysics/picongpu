/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/traits/frame/GetMass.hpp"

namespace picongpu
{
    namespace traits
    {
        namespace attribute
        {
            /** get the mass of a makro particle
             *
             * @param weighting weighting of the particle
             * @param particle a reference to a particle
             *
             * @return mass of the macro particle, sim.unit.mass()
             */
            template<typename T_Particle>
            HDINLINE float_X getMass(float_X const weighting, T_Particle const& particle)
            {
                using ParticleType = T_Particle;
                return picongpu::traits::frame::getMass<typename ParticleType::FrameType>() * weighting;
            }

        } // namespace attribute
    } // namespace traits
} // namespace picongpu
