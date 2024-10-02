/* Copyright 2023 Brian Marre
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

/** @file reset local rate cache sub-stage of atomicPhysics
 *
 * implements the reset of a super cell field shared rate cache for use with atomicPhysics.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalRateCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCache.hpp"
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
    struct ResetLocalRateCache
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_ionSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()() const
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localRateCacheField = *dc.get<
                particles::atomicPhysics::localHelperFields::LocalRateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_localRateCacheField");

            // rate cache inits to all zeros
            localRateCacheField.getDeviceBuffer().setValue(
                picongpu::particles::atomicPhysics::localHelperFields::RateCache<
                    picongpu::traits::GetNumberAtomicStates<IonSpecies>::value>());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
