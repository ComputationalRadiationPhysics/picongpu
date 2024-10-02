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

//! @file resetAcceptedStatus sub-stage of atomicPhysics

#pragma once

#include "picongpu/particles/atomicPhysics/kernel/ResetAcceptedStatus.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

namespace picongpu::particles::atomicPhysics::stage
{
    /** atomic physics sub-stage resetting the macro-ion attribute accepted to false
     *
     * @attention will break an in progress atomicPhysics step, only call at the start or
     *  end of the atomicPhysics step
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct ResetAcceptedStatus
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localTimeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
                "LocalTimeRemainingField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            PMACC_LOCKSTEP_KERNEL(picongpu::particles::atomicPhysics::kernel::ResetAcceptedStatusKernel())
                .config(
                    mapper.getGridDim(),
                    ions)(mapper, localTimeRemainingField.getDeviceDataBox(), ions.getDeviceParticlesBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
