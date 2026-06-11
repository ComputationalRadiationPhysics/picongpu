/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

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
            typename T_SourceParticle,
            typename T_KernelState,
            typename T_Cache,
            typename T_Index,
            typename... T_AdditionalData>
        HDINLINE static T_Number getNumberNewParticles(
            T_Worker const& worker,
            T_SourceParticle& sourceParticle,
            T_KernelState& kernelState,
            T_Cache const& cache,
            T_Index const addtionalDataIndex,
            T_AdditionalData... additionalData);
    };
} // namespace picongpu::particles::creation::moduleInterfaces
