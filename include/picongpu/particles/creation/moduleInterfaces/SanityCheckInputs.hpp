/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/static_assert.hpp>

namespace picongpu::particles::creation::moduleInterfaces
{
    /** interfaces of SanityCheckInputs
     *
     * interface for functor checking T_KernelConfigOptions, additionalData and source-/product-Boxes are consistent
     * with expectations and assumptions.
     *
     * @example check that:
     *   - if T_KernelConfigOptions specifies TransitionType as boundFree, checks that the transitionDataBox passed via
     *     additionalData actually contains boundFree transitions
     *   - the atomicNumbers of the chargeStateDataDataBox and atomicStateDataDataBox passed via additionalData are
     *     consistent
     */
    template<typename T_SourceParticleBox, typename T_ProductParticleBox, typename... T_KernelConfigOptions>
    struct SanityCheckInputs
    {
        //! @returns passes silently if okay
        template<typename T_Index, typename... T_AdditionalData>
        HDINLINE static void validate(
            pmacc::DataSpace<picongpu::simDim> const superCellIndex,
            T_Index const additionalDataIndex,
            T_AdditionalData... additionalData);
    };
} // namespace picongpu::particles::creation::moduleInterfaces
