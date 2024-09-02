/* Copyright 2024 Brian Marre
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

/** @file implements temperature functor interface
 *
 * implementations of this are used for the calculation of a local temperature as ionization potential depression(IPD)
 * input parameter.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    //! interface of functor computing temperature term contribution of particle with given weight and momentum
    struct TemperatureFunctor
    {
        /** calculate term value
         *
         * @param particle
         * @param weightNormalized weight of particle normalized by picongpu::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE
         *
         * @return unit: UNIT_MASS * sim.unit.length()^2 / sim.unit.time()^2 * weight /
         * TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE
         */
        template<typename T_Particle>
        HDINLINE static float_X term(T_Particle& particle, float_64 const weightNormalized);
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
