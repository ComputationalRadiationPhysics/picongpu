/* Copyright 2024 Brian Marre
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

//! @file implements Stewart-Pyatt ionization potential depression(IPD) model

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/IPDModel.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/LocalIPDInputFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/SumFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/ApplyPressureIonization.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/CalculateIPDInput.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/FillIPDSumFields_Electron.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/FillIPDSumFields_Ion.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalFoundUnboundIonField.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics::ionizationPotentialDepression;

    /** implementation of Stewart-Pyatt ionization potential depression(IPD) model
     *
     * @tparam T_TemperatureFunctor term A to average over for all macro particles according to equi-partition theorem,
     * average(A) = k_B * T, must follow
     */
    template<typename T_TemperatureFunctor>
    struct StewartPyattIPD : IPDModel
    {
    private:
        //! reset IPD support infrastructure before we accumulate over particles to calculate new IPD Inputs
        HINLINE static void resetSumFields()
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localSumWeightAllField
                = *dc.get<s_IPD::localHelperFields::SumWeightAllField<picongpu::MappingDesc>>("SumWeightAllField");
            localSumWeightAllField.getDeviceBuffer().setValue(0._X);

            auto& localSumTemperatureFunctionalField
                = *dc.get<s_IPD::localHelperFields::SumTemperatureFunctionalField<picongpu::MappingDesc>>(
                    "SumTemperatureFunctionalField");
            localSumTemperatureFunctionalField.getDeviceBuffer().setValue(0._X);

            auto& localSumWeightElectronField
                = *dc.get<s_IPD::localHelperFields::SumWeightElectronsField<picongpu::MappingDesc>>(
                    "SumWeightElectronsField");
            localSumWeightElectronField.getDeviceBuffer().setValue(0._X);

            auto& localSumChargeNumberIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberIonsField<picongpu::MappingDesc>>(
                    "SumChargeNumberIonsField");
            localSumChargeNumberIonsField.getDeviceBuffer().setValue(0._X);

            auto& localSumChargeNumberSquaredIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberSquaredIonsField<picongpu::MappingDesc>>(
                    "SumChargeNumberSquaredIonsField");
            localSumChargeNumberSquaredIonsField.getDeviceBuffer().setValue(0._X);
        }

    public:
        //! create the sum- and IPD-Input superCell fields required by Stewart-Pyatt
        HINLINE static void createHelperFields(
            picongpu::DataConnector& dataConnector,
            picongpu::MappingDesc const mappingDesc)
        {
            // create sumFields
            //@{
            auto sumWeightAllField
                = std::make_unique<s_IPD::localHelperFields::SumWeightAllField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(sumWeightAllField));
            auto sumTemperatureFunctionalField
                = std::make_unique<s_IPD::localHelperFields::SumTemperatureFunctionalField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumTemperatureFunctionalField));

            auto sumWeightElectronsField
                = std::make_unique<s_IPD::localHelperFields::SumWeightElectronsField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumWeightElectronsField));

            auto sumChargeNumberIonsField
                = std::make_unique<s_IPD::localHelperFields::SumChargeNumberIonsField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumChargeNumberIonsField));
            auto sumChargeNumberSquaredIonsField
                = std::make_unique<s_IPD::localHelperFields::SumChargeNumberSquaredIonsField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumChargeNumberSquaredIonsField));
            //@}

            // create IPD input Fields
            //@{
            // in sim.unit.length(), not weighted
            auto localDebyeLengthField
                = std::make_unique<s_IPD::localHelperFields::LocalDebyeLengthField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(localDebyeLengthField));

            // z^star IPD input field, z^star = = average(q^2) / average(q) ;for q charge number of ion, unitless,
            //  not weighted
            auto localZStarField
                = std::make_unique<s_IPD::localHelperFields::LocalZStarField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(localZStarField));

            // local k_Boltzman * Temperature field, in eV
            auto localTemperatureEnergyField
                = std::make_unique<s_IPD::localHelperFields::LocalTemperatureEnergyField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(localTemperatureEnergyField));
            //@}
        }

        /** calculate all inputs for the ionization potential depression
         *
         * @tparam T_IPDIonSpeciesList list of all species partaking as ions in IPD input
         * @tparam T_IPDElectronSpeciesList list of all species partaking as electrons in IPD input
         *
         * @attention collective over all IPD species
         */
        template<
            uint32_t T_numberAtomicPhysicsIonSpecies,
            typename T_IPDIonSpeciesList,
            typename T_IPDElectronSpeciesList>
        HINLINE static void calculateIPDInput(picongpu::MappingDesc const mappingDesc, uint32_t const)
        {
            using ForEachElectronSpeciesFillSumFields = pmacc::meta::ForEach<
                T_IPDElectronSpeciesList,
                s_IPD::stage::FillIPDSumFields_Electron<boost::mpl::_1, T_TemperatureFunctor>>;
            using ForEachIonSpeciesFillSumFields = pmacc::meta::
                ForEach<T_IPDIonSpeciesList, s_IPD::stage::FillIPDSumFields_Ion<boost::mpl::_1, T_TemperatureFunctor>>;

            // reset IPD SumFields
            resetSumFields();

            ForEachElectronSpeciesFillSumFields{}(mappingDesc);
            ForEachIonSpeciesFillSumFields{}(mappingDesc);

            s_IPD::stage::CalculateIPDInput<T_numberAtomicPhysicsIonSpecies>()(mappingDesc);
        }

        /** check for and apply single step of pressure ionization cascade
         *
         * @attention assumes that ipd-input fields are up to date
         * @attention invalidates ipd-input fields if at least one ionization electron has been spawned
         *
         * @attention must be called once for each step in a pressure ionization cascade
         *
         * @tparam T_AtomicPhysicsIonSpeciesList list of all species partaking as ion in atomicPhysics
         * @tparam T_IPDIonSpeciesList list of all species partaking as ions in IPD input
         * @tparam T_IPDElectronSpeciesList list of all species partaking as electrons in IPD input
         *
         * @attention collective over all ion species
         */
        template<typename T_AtomicPhysicsIonSpeciesList>
        HINLINE static void applyPressureIonization(picongpu::MappingDesc const mappingDesc, uint32_t const)
        {
            using ForEachIonSpeciesApplyPressureIonization = pmacc::meta::ForEach<
                T_AtomicPhysicsIonSpeciesList,
                s_IPD::stage::ApplyPressureIonization<boost::mpl::_1, StewartPyattIPD<T_TemperatureFunctor>>>;

            ForEachIonSpeciesApplyPressureIonization{}(mappingDesc);
        };

        /** calculate ionization potential depression
         *
         * @param localTemperatureEnergyBox deviceDataBox giving access to the local temperature * k_Boltzman for all
         *  local superCells, in sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2, not weighted
         * @param localZStarBox deviceDataBox giving access to the local z^Star value, = average(q^2) / average(q),
         *  for all local superCells, unitless, not weighted
         * @param localDebyeLengthBox deviceDataBox giving access to the local debye length for all local superCells,
         *  sim.unit.length(), not weighted
         * @param superCellFieldIdx index of superCell in superCellField(without guards)
         *
         * @return unit: eV, not weighted
         */
        template<
            uint8_t T_atomicNumber,
            typename T_LocalDebyeLengthBox,
            typename T_LocalTemperatureEnergyBox,
            typename T_LocalZStarBox>
        HDINLINE static float_X calculateIPD(
            pmacc::DataSpace<simDim> const superCellFieldIdx,
            T_LocalDebyeLengthBox const localDebyeLengthBox,
            T_LocalTemperatureEnergyBox const localTemperatureEnergyBox,
            T_LocalZStarBox const localZStarBox)
        {
            // eV/(sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2)
            constexpr float_X eV = static_cast<float_X>(
                picongpu::sim.unit.mass() * pmacc::math::cPow(picongpu::sim.unit.length(), 2u)
                / pmacc::math::cPow(picongpu::sim.unit.time(), 2u) * picongpu::UNITCONV_Joule_to_keV * 1e3);

            // eV/(sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2) * unitless * sim.unit.charge()^2
            //  / ( unitless * sim.unit.charge()^2 * sim.unit.time()^2 / (sim.unit.length()^3 * sim.unit.mass()))
            // = eV * sim.unit.time()^2 * sim.unit.mass()^(-1) * sim.unit.length()^(-2) * sim.unit.charge()^2 *
            // sim.unit.charge()^(-2)
            //  * sim.unit.time()^(-2) * sim.unit.length()^3 * sim.unit.mass()^1 = eV * sim.unit.length()
            // eV * sim.unit.length()
            constexpr float_X constFactor = eV * static_cast<float_X>(T_atomicNumber)
                * pmacc::math::cPow(picongpu::sim.pic.getElectronCharge(), 2u)
                / (4._X * static_cast<float_X>(picongpu::PI) * picongpu::sim.pic.getEps0());

            // eV, not weighted
            float_X const temperatureTimesk_Boltzman = localTemperatureEnergyBox(superCellFieldIdx);
            // sim.unit.length(), not weighted
            float_X const debyeLength = localDebyeLengthBox(superCellFieldIdx);

            // (eV * sim.unit.length()) / (eV * sim.unit.length()), not weighted
            // unitless, not weighted
            float_X const K = constFactor / (temperatureTimesk_Boltzman * debyeLength);

            // unitless, not weighted
            float_X const zStar = localZStarBox(superCellFieldIdx);

            // eV, not weighted
            return temperatureTimesk_Boltzman * (math::pow(((3 * zStar + 1) * K + 1), 2._X / 3._X) - 1._X)
                / (2._X * (zStar + 1._X));
        }

        template<typename T_Kernel, uint32_t T_chunkSize, typename... T_KernelInput>
        HINLINE static void callKernelWithIPDInput(
            pmacc::DataConnector& dc,
            pmacc::AreaMapping<CORE + BORDER, picongpu::MappingDesc>& mapper,
            T_KernelInput... kernelInput)
        {
            auto& localDebyeLengthField
                = *dc.get<s_IPD::localHelperFields::LocalDebyeLengthField<picongpu::MappingDesc>>(
                    "LocalDebyeLengthField");
            auto& localTemperatureEnergyField
                = *dc.get<s_IPD::localHelperFields::LocalTemperatureEnergyField<picongpu::MappingDesc>>(
                    "LocalTemperatureEnergyField");
            auto& localZStarField
                = *dc.get<s_IPD::localHelperFields::LocalZStarField<picongpu::MappingDesc>>("LocalZStarField");

            PMACC_LOCKSTEP_KERNEL(T_Kernel())
                .template config<T_chunkSize>(mapper.getGridDim())(
                    mapper,
                    kernelInput...,
                    localDebyeLengthField.getDeviceDataBox(),
                    localTemperatureEnergyField.getDeviceDataBox(),
                    localZStarField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
