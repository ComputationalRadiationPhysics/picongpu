/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file reset local rate cache sub-stage of atomicPhysics
 *
 * implements the reset of a super cell field shared rate cache for use with atomicPhysics.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCache.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCacheField.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/traits/GetNumberAtomicStates.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** @class atomic physics sub-stage for a species
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct ResetRateCache
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_ionSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()() const
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& rateCacheField = *dc.get<
                particles::atomicPhysics::localHelperFields::RateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_rateCacheField");

            // rate cache inits to all zeros
            rateCacheField.getDeviceBuffer().setValue(
                picongpu::particles::atomicPhysics::localHelperFields::RateCache<
                    picongpu::traits::GetNumberAtomicStates<IonSpecies>::value>());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
