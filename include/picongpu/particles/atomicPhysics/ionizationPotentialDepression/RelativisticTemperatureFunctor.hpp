/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
    //! functor computing relativistic temperature contribution of particle with given weight and momentum
    struct RelativisticTemperatureFunctor : TemperatureFunctor
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
            float3_64 const momentumVector = static_cast<float3_64>(particle[momentum_]);

            // sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2 * weight^2 /
            // sim.unit.typicalNumParticlesPerMacroParticle()^2
            float_64 const momentumSquared
                = pmacc::math::l2norm2(momentumVector)
                  / pmacc::math::cPow(picongpu::sim.unit.typicalNumParticlesPerMacroParticle(), 2u);

            // sim.unit.mass(), not weighted
            float_64 const mass
                = static_cast<float_64>(picongpu::traits::frame::getMass<typename T_Particle::FrameType>());
            // sim.unit.length() / sim.unit.time(), not weighted
            constexpr float_64 c = picongpu::sim.pic.getSpeedOfLight();
            // sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2, not weighted
            float_64 const m2c2 = pmacc::math::cPow(mass * c, 2u);

            // sim.unit.length() / sim.unit.time()
            //  * (sim.unit.mass()^2 * sim.unit.length()^2 / sim.unit.time()^2 * weight^2 /
            //  sim.unit.typicalNumParticlesPerMacroParticle()^2) / sqrt((sim.unit.mass()^2 * sim.unit.length()^2 /
            //  sim.unit.time()^2 * weight^2
            //      / sim.unit.typicalNumParticlesPerMacroParticle()^2) + sim.unit.mass()^2 * sim.unit.length()^2 /
            //      sim.unit.time()^2 * weight^2 / sim.unit.typicalNumParticlesPerMacroParticle()^2)
            // = sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2 * weight /
            // sim.unit.typicalNumParticlesPerMacroParticle()
            return /*since we sum over all three dimensions */ (1._X / 3._X)
                   * static_cast<float_X>(
                       c * momentumSquared
                       / math::sqrt(momentumSquared + m2c2 * pmacc::math::cPow(weightNormalized, 2u)));
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
