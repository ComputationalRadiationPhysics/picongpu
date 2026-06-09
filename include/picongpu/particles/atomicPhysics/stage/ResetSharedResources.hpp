/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2023-2024 Brian Marre
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

//! @file reset deltaWeight entry of all histogram bin

#pragma once

// need picongpu::atomicPhysics::ElectronHistogram from atomicPhysics.param
#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/kernel/ResetDeltaWeightElectronHistogram.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/FieldEnergyUseCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics/param.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::stage
{
    /// short hand for
    namespace localHelperFields = picongpu::particles::atomicPhysics::localHelperFields;

    /** ResetSharedRessources atomic physics sub-stage
     *
     * reset all superCell-shared atomicPhysics-transition's resource usage tracking, such as:
     * - deltaWeight entries of electron histogram bin
     * - FieldEnergyUseCache entry for each cell of each superCell
     *
     * @tparam T_numberAtomicPhysicsIonSpecies specialization template parameter used to prevent compilation of all
     *  atomicPhysics kernels if no atomic physics species is present.
     */
    template<uint32_t T_numberAtomicPhysicsIonSpecies>
    struct ResetSharedResources
    {
        using Mapping = picongpu::MappingDesc;

        HINLINE static void resetFieldEnergyUseCache(pmacc::DataConnector& dc)
        {
            using FieldEnergyUseCacheField = localHelperFields::FieldEnergyUseCacheField<Mapping>;

            auto& fieldEnergyUseCacheField = *dc.get<FieldEnergyUseCacheField>("FieldEnergyUseCacheField");
            fieldEnergyUseCacheField.getDeviceBuffer().setValue(
                localHelperFields::
                    FieldEnergyUseCache<FieldEnergyUseCacheField::Extent, FieldEnergyUseCacheField::StorageType>());
        }

        HINLINE static void resetElectronHistogramDeltaWeight(
            pmacc::DataConnector& dc,
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> const mapper)
        {
            auto& timeRemainingField = *dc.get<localHelperFields::TimeRemainingField<Mapping>>("TimeRemainingField");

            /** @note The better readable version:
             *
             * @code{.cpp}
             * using ElectronHistogram = typename picongpu::atomicPhysics::ElectronHistogram;
             * @endcode
             *
             * causes a spurious compiler warning in gcc-9 and gcc-10 as discussed here,
             * https://gcc.gnu.org/pipermail/gcc-patches/2020-May/546603.html,
             */
            using picongpu::atomicPhysics::ElectronHistogram;

            auto& electronHistogramField = *dc.get<
                particles::atomicPhysics::electronDistribution::LocalHistogramField<ElectronHistogram, Mapping>>(
                "Electron_HistogramField");

            // macro for call of kernel for every superCell
            PMACC_LOCKSTEP_KERNEL(
                picongpu::particles::atomicPhysics::kernel::
                    ResetDeltaWeightElectronHistogramKernel<ElectronHistogram, T_numberAtomicPhysicsIonSpecies>())
                .template config<ElectronHistogram::numberBins>(mapper.getGridDim())(
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    electronHistogramField.getDeviceDataBox());
        }

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            /// @todo implement photon histogram, Brian Marre, 2023
            resetFieldEnergyUseCache(dc);
            resetElectronHistogramDeltaWeight(dc, mapper);
        }
    };

    //! specialization for no atomicPhysics species in simulation
    template<>
    struct ResetSharedResources<0u>
    {
        HINLINE void operator()([[maybe_unused]] picongpu::MappingDesc const mappingDesc) const
        {
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
