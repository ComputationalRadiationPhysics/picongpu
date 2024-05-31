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
     * @details see files included by picongpu/particles/creation/SpawnFromSourceSpeciesModuleInterfaces.hpp for
     *  interface definitions.
     *
     * @tparam T_SanityCheckInputs functor checking T_KernelConfigOptions, additionalData and source-/product- Boxes
     * are consistent with expectations and assumptions. e.g. check that:
     *   - if T_KernelConfigOptions specifies TransitionType as boundFree, checks that the transitionDataBox passed via
     *     additionalData actually contains boundFree transitions
     *   - the atomicNumbers of the chargeStateDataDataBox and atomicStateDataDataBox passed via additionalData are
     *     consistent
     * @tparam T_SuperCellFilterFunctor test allowing entire superCell to be skipped depending on additionalData or
     * superCell index
     * e.g. skip superCell if localTimeRemainingDataBox[additionalDataIndex] is > 0
     * @tparam T_PredictorFunctor functor predicting number of product species particles to spawn for a given source
     * species particle, depending on kernelState and additionalData.
     * @note may update source particle!
     * @tparam T_ParticlePairUpdateFunctor functor initialising spawned productSpecies particle based on
     * additionalData and sourceSpecies particle
     * @tparam T_KernelStateType type of kernelState, one instance for each superCell
     * @tparam T_InitKernelStateFunctor functor initialising T_KernelStateType variable
     * @tparam T_AdditionalDataIndexFunctor functor returning index to access additionalData by
     *  @note only one is supported for all additionalData
     *  @note dimension is configurable
     *  @note may be ignored for some or all additionalData
     * @tparam T_WriteOutKernelStateFunctor write out something based on the final kernelState to additionalData
     */
    template<
        template<
            typename T_SourceParticleBox,
            typename T_ProductParticleBox,
            typename... T_KernelConfigOptionsAndAdditionalData>
        typename T_SanityCheckInputs,
        template<typename... T_KernelConfigOptions>
        typename T_SuperCellFilterFunctor,
        template<typename T_Number, typename... T_KernelConfigOptions>
        typename T_PredictorFunctor,
        template<typename... T_KernelConfigOptions>
        typename T_ParticlePairUpdateFunctor,
        typename T_KernelStateType,
        template<typename... T_KernelConfigOptions>
        typename T_InitKernelStateFunctor,
        template<typename... T_KernelConfigOptions>
        typename T_AdditionalDataIndexFunctor,
        template<typename... T_KernelConfigOptions>
        typename T_WriteOutKernelStateFunctor>
    struct ModuleConfig
    {
        template<
            typename T_SourceParticleBox,
            typename T_ProductParticleBox,
            typename... T_KernelConfigOptionsAndAdditionalData>
        using SanityCheckInputs = T_SanityCheckInputs<
            T_SourceParticleBox,
            T_ProductParticleBox,
            T_KernelConfigOptionsAndAdditionalData...>;

        template<typename... T_KernelConfigOptions>
        using SuperCellFilterFunctor = T_SuperCellFilterFunctor<T_KernelConfigOptions...>;

        template<typename T_Number, typename... T_KernelConfigOptions>
        using PredictorFunctor = T_PredictorFunctor<T_Number, T_KernelConfigOptions...>;

        template<typename... T_KernelConfigOptions>
        using ParticlePairUpdateFunctor = T_ParticlePairUpdateFunctor<T_KernelConfigOptions...>;

        using KernelStateType = T_KernelStateType;

        template<typename... T_KernelConfigOptions>
        using AdditionalDataIndexFunctor = T_AdditionalDataIndexFunctor<T_KernelConfigOptions...>;

        template<typename... T_KernelConfigOptions>
        using InitKernelStateFunctor = T_InitKernelStateFunctor<T_KernelConfigOptions...>;

        template<typename... T_KernelConfigOptions>
        using WriteOutKernelStateFunctor = T_WriteOutKernelStateFunctor<T_KernelConfigOptions...>;
    };
} // namespace picongpu::particles::creation
