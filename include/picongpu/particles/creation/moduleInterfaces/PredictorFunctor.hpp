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
    /** interface of PredictorFunctor
     *
     * functor predicting number of product species particles to spawn for a given source species particle,
     * depending on passed kernelState and additionalData
     *
     * @note may update source particle!
     */
    template<typename T_Number, typename... T_KernelConfigOptions>
    struct PredictorFunctor
    {
        template<
            typename T_Worker,
            typename T_Index,
            typename T_SourceParticle,
            typename T_KernelState,
            typename... T_AdditionalData>
        HDINLINE static T_Number getNumberNewParticles(
            T_Worker const& worker,
            T_SourceParticle& sourceParticle,
            T_KernelState& kernelState,
            T_Index const addtionalDataIndex,
            T_AdditionalData... additionalData);
    };
} // namespace picongpu::particles::creation::moduleInterfaces
