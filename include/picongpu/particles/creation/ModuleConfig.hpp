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

namespace picongpu::particles::creation
{
    /** configuration wrapper, holding all modules to be used by the SpawnFromSourceSpecies kernel framework
     *
     * @details see picongpu/particles/creation/SpawnFromSourceSpeciesModuleInterfaces.hpp for interface definitions.
     *
     * @tparam T_SanityCheckInputs struct containing compile time asserts checking settings in T_ConfigOptions and
     *  T_SharedDataBoxes are consistent with expectations and assumptions
     *  e.g. check that:
     *   - if T_ConfigOptions specifies TransitionType as boundFree the transitionDataBox passed via T_SharedDataBoxes
     *     should actually contains boundFree transitions
     *   - the atomicNumbers of the chargeStateDataDataBox and atomicStateDataDataBox passed via T_SharedDataBoxes are
     *     consistent
     * @tparam T_SkipSuperCellFunctor test allowing entire superCell to be skipped depending on sharedDataBoxes,
     *  @note to skip test, use empty function
     *  e.g. skip superCell if localTimeRemainingDataBox[sharedDataBoxIndex] is > 0
     * @tparam T_PredictorFunctor functor predicting number of product species particles to spawn for a given source
     *  species particle, depending on sharedState, @note may/(may not) update source particle!
     * @tparam T_ParticlePairUpdateFunctor functor initialising spawned productSpecies particle based on
     * sharedDataBoxes and sourceSpecies particle
     * @tparam T_SharedStateType type of sharedState, one instance for each superCell
     * @tparam T_InitSharedStateFunctor functor initialising sharedState variable
     * @tparam T_SharedDataBoxIndexFunctor functor returning index to access sharedDataBoxes by,
     *  @note only one is supported for all sharedDataBoxes
     *  @note dimension is configurable
     *  @note may be ignored for some or all sharedDataBoxes
     * @tparam T_WriteOutSharedStateFunctor write out final sharedState
     */
    template<
        template<
            typename T_SourceParticleBox,
            typename T_ProductParticleBox,
            typename... T_KernelConfigOptionsAndSharedDataBoxTypes>
        typename T_SanityCheckInputs,
        template<typename... T_KernelConfigOptions>
        typename T_SkipSuperCellFunctor,
        template<typename T_Number, typename... T_KernelConfigOptions>
        typename T_PredictorFunctor,
        typename T_ParticlePairUpdateFunctor,
        typename T_SharedStateType,
        template<typename... T_KernelConfigOptions>
        typename T_InitSharedStateFunctor,
        template<typename... T_KernelConfigOptions>
        typename T_SharedDataBoxIndexFunctor,
        template<typename... T_KernelConfigOptions>
        typename T_WriteOutSharedStateFunctor>
    struct ModuleConfig
    {
        template<
            typename T_SourceParticleBox,
            typename T_ProductParticleBox,
            typename... T_KernelConfigOptionsAndSharedDataBoxTypes>
        using SanityCheckInputs = T_SanityCheckInputs<
            T_SourceParticleBox,
            T_ProductParticleBox,
            T_KernelConfigOptionsAndSharedDataBoxTypes...>;

        template<typename... T_KernelConfigOptions>
        using SkipSuperCellFunctor = T_SkipSuperCellFunctor<T_KernelConfigOptions...>;

        template<typename T_Number, typename... T_KernelConfigOptions>
        using PredictorFunctor = T_PredictorFunctor<T_Number, T_KernelConfigOptions...>;

        using ParticlePairUpdateFunctor = T_ParticlePairUpdateFunctor;
        using SharedStateType = T_SharedStateType;

        template<typename... T_KernelConfigOptions>
        using SharedDataBoxIndexFunctor = T_SharedDataBoxIndexFunctor<T_KernelConfigOptions...>;

        template<typename... T_KernelConfigOptions>
        using InitSharedStateFunctor = T_InitSharedStateFunctor<T_KernelConfigOptions...>;

        template<typename... T_KernelConfigOptions>
        using WriteOutSharedStateFunctor = T_WriteOutSharedStateFunctor<T_KernelConfigOptions...>;
    };
} // namespace picongpu::particles::creation
