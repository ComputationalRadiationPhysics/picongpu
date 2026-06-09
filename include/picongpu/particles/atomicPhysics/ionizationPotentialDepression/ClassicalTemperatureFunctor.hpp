/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2024-2024 Brian Marre
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

/** @file implements relativistic temperature term to average over
 *
 * is used for the calculation of a local temperature as ionization potential depression(IPD) input parameter.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/TemperatureFunctor.hpp"
#include "picongpu/traits/frame/GetMass.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    //! functor computing non-relativistic temperature contribution of particle with given weight and momentum
    struct ClassicalTemperatureFunctor : TemperatureFunctor
    {
        /** calculate term value for given particle
         *
         * @param particle
         * @param weightNormalized weight of particle normalized by
         * picongpu::sim.unit.typicalNumParticlesPerMacroParticle()
         *
         * @return unit: sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2 * weight /
         * sim.unit.typicalNumParticlesPerMacroParticle()
         */
        template<typename T_Particle>
        HDINLINE static float_X term(T_Particle& particle, float_64 const weightNormalized)
        {
            // sim.unit.mass() * sim.unit.length() / sim.unit.time() * weight
            float3_64 const momentumVectorNormalized = static_cast<float3_64>(particle[momentum_]);

            // sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2 * weight^2 /
            // sim.unit.typicalNumParticlesPerMacroParticle()^2
            float_64 momentumSquared
                = pmacc::math::l2norm2(momentumVectorNormalized)
                  / pmacc::math::cPow(picongpu::sim.unit.typicalNumParticlesPerMacroParticle(), 2u);

            // sim.unit.mass(), not weighted
            float_64 const mass
                = static_cast<float_64>(picongpu::traits::frame::getMass<typename T_Particle::FrameType>());

            // (sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2 * weight^2 /
            // sim.unit.typicalNumParticlesPerMacroParticle()^2)
            //  / (sim.unit.mass() * weight / sim.unit.typicalNumParticlesPerMacroParticle())
            // sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2 * weight /
            // sim.unit.typicalNumParticlesPerMacroParticle()
            // unit_energy * weight / sim.unit.typicalNumParticlesPerMacroParticle()
            return (2._X / 3._X) * static_cast<float_X>(momentumSquared / (2.0 * mass * weightNormalized));
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
