/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2024-2024 Brian Marre
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

//! @file update electric field with field ionization energy use

#pragma once

// need picongpu::atomicPhysics::ElectronHistogram from atomicPhysics.param
#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/particles/atomicPhysics/kernel/UpdateElectricField.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/FieldEnergyUseCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::stage
{
    /** UpdateElectricField atomic physics sub-stage
     *
     * remove the used field energy from the electric field by reducing the norm of the local E-Field vector to match
     *  the energy use
     *
     * @tparam T_numberAtomicPhysicsIonSpecies specialization template parameter used to prevent compilation of all
     *  atomicPhysics kernels if no atomic physics species is present.
     */
    template<uint32_t T_numberAtomicPhysicsIonSpecies>
    struct UpdateElectricField
    {
        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& timeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");

            auto& eField = *dc.get<FieldE>(FieldE::getName());

            using FieldEnergyUseCacheField
                = picongpu::particles::atomicPhysics::localHelperFields::FieldEnergyUseCacheField<
                    picongpu::MappingDesc>;
            auto& fieldEnergyUseCacheField = *dc.get<FieldEnergyUseCacheField>("FieldEnergyUseCacheField");

            // macro for call of kernel for every superCell, see pull request #4321
            PMACC_LOCKSTEP_KERNEL(
                particles::atomicPhysics::kernel::UpdateElectricFieldKernel<T_numberAtomicPhysicsIonSpecies>())
                .template config<FieldEnergyUseCacheField::ValueType::numberCells>(mapper.getGridDim())(
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    eField.getDeviceDataBox(),
                    fieldEnergyUseCacheField.getDeviceDataBox());
        }
    };

    //! specialization for no atomicPhysics ion species
    template<>
    struct UpdateElectricField<0u>
    {
        HINLINE void operator()([[maybe_unused]] picongpu::MappingDesc const mappingDesc) const
        {
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
