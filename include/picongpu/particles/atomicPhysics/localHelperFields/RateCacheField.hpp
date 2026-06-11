/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCache.hpp"
#include "picongpu/particles/traits/GetNumberAtomicStates.hpp"

#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /**@class superCell field of the no-change-transition rateCache
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     * @tparam T_IonSpecies resolved type of ion species
     */
    template<typename T_MappingDescription, typename T_IonSpecies>
    struct RateCacheField
        : public SuperCellField<
              RateCache<picongpu::traits::GetNumberAtomicStates<T_IonSpecies>::value>,
              T_MappingDescription,
              false /*no guards*/>
    {
        using FrameType = typename T_IonSpecies::FrameType;

        RateCacheField(T_MappingDescription const& mappingDesc)
            : SuperCellField<
                  RateCache<picongpu::traits::GetNumberAtomicStates<T_IonSpecies>::value>,
                  T_MappingDescription,
                  false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return FrameType::getName() + "_rateCacheField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
