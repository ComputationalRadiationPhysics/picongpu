/* Copyright 2022-2023 Brian Marre
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

/** @file implements a super cell local cache of no-change transition rates for each
 *      atomic state of a species.
 *
 * no-change atomic physics transition rates(diagonal elements of rate matrix) are expensive
 *  to calculate and all have to be calculated anyway for the adaptive time step calculation.
 *
 * Therefore the are only calculated for all atomic states once per atomicPhysics substep
 *  and cached for use in the rate solver.
 *
 * Since no-change transition rates depend on the local electron spectrum, as well as all
 *  transition's parameters, they are super cell local, same as the electron spectrum.
 */

#pragma once

#include "picongpu/particles/atomicPhysics2/SuperCellField.hpp"
#include "picongpu/particles/atomicPhysics2/localHelperFields/RateCache.hpp"
#include "picongpu/particles/traits/GetNumberAtomicStates.hpp"

#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2::localHelperFields
{
    /**@class superCell field of the no-change-transition rateCache
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     * @tparam T_IonSpecies resolved type of ion species
     */
    template<typename T_MappingDescription, typename T_IonSpecies>
    struct LocalRateCacheField
        : public SuperCellField<
              RateCache<picongpu::traits::GetNumberAtomicStates<T_IonSpecies>::value>,
              T_MappingDescription,
              false /*no guards*/>
    {
        using FrameType = typename T_IonSpecies::FrameType;

        LocalRateCacheField(T_MappingDescription const& mappingDesc)
            : SuperCellField<
                RateCache<picongpu::traits::GetNumberAtomicStates<T_IonSpecies>::value>,
                T_MappingDescription,
                false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return FrameType::getName() + "_localRateCacheField";
        }
    };
} // namespace picongpu::particles::atomicPhysics2::localHelperFields
