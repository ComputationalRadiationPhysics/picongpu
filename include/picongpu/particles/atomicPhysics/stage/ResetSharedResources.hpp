/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
