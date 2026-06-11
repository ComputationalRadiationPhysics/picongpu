/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file implements temperature functor interface
 *
 * implementations of this are used for the calculation of a local temperature as ionization potential depression(IPD)
 * input parameter.
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    //! interface of functor computing temperature term contribution of particle with given weight and momentum
    struct TemperatureFunctor
    {
        /** calculate term value
         *
         * @param particle
         * @param weightNormalized weight of particle normalized by
         * picongpu::sim.unit.typicalNumParticlesPerMacroParticle()
         *
         * @return unit: sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2 * weight /
         * sim.unit.typicalNumParticlesPerMacroParticle()
         */
        template<typename T_Particle>
        HDINLINE static float_X term(T_Particle& particle, float_64 const weightNormalized);
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
