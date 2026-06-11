/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file CalculateIPDInput ionization potential depression(IPD) sub-stage
 *
 * implements calculation of IPD input parameters from the local sumField values
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/LocalIPDInputFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/SumFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/kernel/CalculateIPDInput.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"

#include <string>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stage
{
    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics::ionizationPotentialDepression;

    /** IPD sub-stage for calculating IPD input from sumFields, required for calculating IPD
     *
     * @tparam T_numberAtomicPhysicsIonSpecies specialization template parameter used to prevent compilation of all
     *  atomicPhysics kernels if no atomic physics species is present.
     */
    template<uint32_t T_numberAtomicPhysicsIonSpecies>
    struct CalculateIPDInput
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

            auto& localSumWeightAllField
                = *dc.get<s_IPD::localHelperFields::SumWeightAllField<picongpu::MappingDesc>>("SumWeightAllField");
            auto& localSumTemperatureFunctionalField
                = *dc.get<s_IPD::localHelperFields::SumTemperatureFunctionalField<picongpu::MappingDesc>>(
                    "SumTemperatureFunctionalField");

            auto& localSumWeightElectronField
                = *dc.get<s_IPD::localHelperFields::SumWeightElectronsField<picongpu::MappingDesc>>(
                    "SumWeightElectronsField");

            auto& localSumChargeNumberIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberIonsField<picongpu::MappingDesc>>(
                    "SumChargeNumberIonsField");
            auto& localSumChargeNumberSquaredIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberSquaredIonsField<picongpu::MappingDesc>>(
                    "SumChargeNumberSquaredIonsField");

            auto& temperatureEnergyField
                = *dc.get<s_IPD::localHelperFields::TemperatureEnergyField<picongpu::MappingDesc>>(
                    "TemperatureEnergyField");
            auto& zStarField = *dc.get<s_IPD::localHelperFields::ZStarField<picongpu::MappingDesc>>("ZStarField");
            auto& debyeLengthField
                = *dc.get<s_IPD::localHelperFields::DebyeLengthField<picongpu::MappingDesc>>("DebyeLengthField");
            auto& freeElectronDensityField
                = *dc.get<s_IPD::localHelperFields::ZStarField<picongpu::MappingDesc>>("FreeElectronDensityField");

            // macro for kernel call
            PMACC_LOCKSTEP_KERNEL(s_IPD::kernel::CalculateIPDInputKernel<T_numberAtomicPhysicsIonSpecies>())
                .template config<1u>(mapper.getGridDim())(
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    localSumWeightAllField.getDeviceDataBox(),
                    localSumTemperatureFunctionalField.getDeviceDataBox(),
                    localSumWeightElectronField.getDeviceDataBox(),
                    localSumChargeNumberIonsField.getDeviceDataBox(),
                    localSumChargeNumberSquaredIonsField.getDeviceDataBox(),
                    debyeLengthField.getDeviceDataBox(),
                    temperatureEnergyField.getDeviceDataBox(),
                    zStarField.getDeviceDataBox(),
                    freeElectronDensityField.getDeviceDataBox());
        }
    };

    //! specialization for no atomicPhysics ion species
    template<>
    struct CalculateIPDInput<0u>
    {
        //! call of kernel for every superCell
        HINLINE void operator()([[maybe_unused]] picongpu::MappingDesc const mappingDesc) const
        {
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stage
