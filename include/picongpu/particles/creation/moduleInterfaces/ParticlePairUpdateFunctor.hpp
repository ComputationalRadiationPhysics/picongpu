/* Copyright 2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software you can redistribute it and or modify
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

#pragma once

#include "picongpu/simulation_defines.hpp"

namespace picongpu::particles::creation::moduleInterfaces
{
    //! interface of ParticlePairUpdateFunctor
    template<typename... T_KernelConfigOptions>
    struct ParticlePairUpdateFunctor
    {
        /** functor initialising spawned productSpecies particle based on additionalData and sourceSpecies particle
         *
         * @param worker object containing the device and block information
         * @param cascadeIndex index of product particle to be initialized from the specific given source particle,
         *  \in [1, predictorResult]
         *  @note may be used to initialize particles differently depending on index
         *  @example ionization cascade, create electrons with differing energies depending on position in cascade
         *
         * @note may also update source particle attributes!
         */
        template<
            typename T_Worker,
            typename T_SourceParticle,
            typename T_ProductParticle,
            typename T_Number,
            typename T_KernelStateType,
            typename T_Index,
            typename... T_AdditionalData>
        HDINLINE static void update(
            T_Worker const& worker,
            T_SourceParticle& sourceParticle,
            T_ProductParticle& productParticle,
            IdGenerator& idGen,
            T_Number const cascadeIndex,
            T_KernelStateType& kernelState,
            T_Index const additionalDataIndex,
            T_AdditionalData... additionalData);
    };
} // namespace picongpu::particles::creation::moduleInterfaces
