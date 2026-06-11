/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file implements Stewart-Pyatt ionization potential depression(IPD) model

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/IPDModel.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/LocalIPDInputFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/SumFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/ApplyIPDIonization.def"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/CalculateIPDInput.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/FillIPDSumFields_Electron.def"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stage/FillIPDSumFields_Ion.def"
#include "picongpu/particles/atomicPhysics/localHelperFields/FoundUnboundIonField.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    namespace detail
    {
        struct StewartPyattSuperCellConstantInput
        {
            // eV
            float_X temperatureTimesk_Boltzman;
            // UNIT_LENGTH
            float_X debyeLength;
            // unitless
            float_X zStar;
            // 1/unit_length^3
            float_X freeElectronDensity;

            constexpr StewartPyattSuperCellConstantInput(
                float_X const temperatureTimesk_Boltzman_i,
                float_X const debyeLength_i,
                float_X const zStar_i,
                float_X const freeElectronDensity_i)
                : temperatureTimesk_Boltzman(temperatureTimesk_Boltzman_i)
                , debyeLength(debyeLength_i)
                , zStar(zStar_i)
                , freeElectronDensity(freeElectronDensity_i)
            {
            }
        };
    } // namespace detail

    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics::ionizationPotentialDepression;

    /** implementation of Stewart-Pyatt ionization potential depression(IPD) model
     *
     * @tparam T_TemperatureFunctor functor returning for a macro particle some value A, with average(A) = k_B * T
     *  according to equi-partition theorem
     * @tparam T_useSCFLYextendedStewartPyattIPD true =^= use minimum of Stewart-Pyatt and additional SCFLY IPD branch,
     *                                           false =^= use Stewart-Pyatt IPD branch directly
     *
     * @details The implementation of the Stewart-Pyatt IPD model is based on [1],  with the choice of central core
     *  charge as (chargeState + 1) according to [2] and the additional IPD branch implemented in SCFLY [3].
     *
     * @details additional SCFLY IPD-branch,
     *  The SCFLY implementation of the Stewart-Pyatt IPD-model has an additional branch not mentioned in the
     *  original SCFLY publication [4], the preceding FLYCHK publication [5], or the later description of the
     *  extended IPD model in [2].
     *  The SCFLY Stewart-Pyatt IPD implementation uses the minimum of this additional branch and the actual
     *  Stewart-Pyatt IPD-model.
     *  I was unable to find a source for this additional branch, Brian Marre, 2025.
     *  Therefore the implementation of this branch is based directly on the non-public SCFLY source code.
     *
     * @details SCFLY source code for the IPD-implementation,
     *  - actual IPD implementation, function debye,
     * https://github.com/ComputationalRadiationPhysics/scfly/blob/5d9b0a997bb6688b04e1a34a7fa8db0f2657106a/code/src/scdrv.f#L3519-L3547
     *  - called in the subroutine depress,
     * https://github.com/ComputationalRadiationPhysics/scfly/blob/5d9b0a997bb6688b04e1a34a7fa8db0f2657106a/code/src/scdrv.f#L3310-L3517
     *
     * @details for code comparisons between the FLYonPIC and SCFLY Stewart-Pyatt IPD model,
     *  The SCFLY implementation uses the secondary form of the Stewart-Pyatt IPD-Model, described in [1], below
     *  equation (5).
     * To get this form, assume (zStar == chargeState), ie. the plasma consists of only one ion, with only one charge
     *  state,
     * replace (zStar + 1) with the debye length definition including ions and electrons,
     *    (zStar + 1) = 1/(lambda_Debye^2) * epsilon_0 * k_B * T /(e^2 * n_e)
     *      lambda_Debye ... debye length
     *      epsilon_0    ... vacuum permittivity
     *      k_B          ... boltzman constant
     *      n_e          ... free electron density
     *      e            ... elementary charge
     *      zStar = average(z^2)/average(z) ... z is charge state of an ion
     * substitute this into the term,
     *    3 * (zStar + 1) * K
     * use the definition of K and and identify the remaining factors as a power of the ion sphere radius,
     *    3 * (zStar+1) * K = ionSphereRadius^3 / lambda_Debye^3
     * then substitute the resulting expression back into equation (5) of Stewart-Pyatt, which gives the term used in
     * the SCFLY implementation,
     * https://github.com/ComputationalRadiationPhysics/scfly/blob/5d9b0a997bb6688b04e1a34a7fa8db0f2657106a/code/src/scdrv.f#L3537-L3538
     *
     * References:
     * - [1] "Lowering of Ionization Potentials in Plasmas", John C. Stewart and Kedat D. Pyatt Jr.,
     *    Astrophysical Journal, vol. 144 (1966), p.1203-1211, , specifically equation (5) on p. 1208
     * - [2] "The effects of ionization potential depression on the spectra emitted by hot dense aluminium plasmas",
     *    Thomas R. Preston, Sam. M. Vinko, Orlando Ciricosta, Hyun-Kyung Chung, Richard W. Lee, Justin S. Wark
     *    High Energy Density Physics 9 (2013) 258-263, equation 1 on p. 259
     * - [3] SCFLY IPD-model source code,
     * https://github.com/ComputationalRadiationPhysics/scfly/blob/5d9b0a997bb6688b04e1a34a7fa8db0f2657106a/code/src/scdrv.f#L3348-L3349
     * - [4] "Extension of atomic configuration sets of the Non-LTE model in the application to the K-\alpha
     *        diagnostics of hot dense matter",
     *    H.K. CHung, M.H. Chen, R.W. Lee,
     *    High energy Density Physics 3 (2007) 57-64
     * - [5] "FLYCHK: Generalized population kinetics and spectral model for rapid spectroscopy for all elements",
     *    H.-K. Chung, M.H. Chen, W.L. Morgan, Y. Ralchenko, R.W. Lee,
     *    High Energy Density Physics 1 (2003) 3-12
     */
    template<typename T_TemperatureFunctor, bool T_useSCFLYextendedStewartPyattIPD>
    struct StewartPyattIPD : IPDModel
    {
        using SuperCellConstantInput = detail::StewartPyattSuperCellConstantInput;

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
            auto debyeLengthField
                = std::make_unique<s_IPD::localHelperFields::DebyeLengthField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(debyeLengthField));

            // local k_Boltzman * Temperature field, in eV
            auto temperatureEnergyField
                = std::make_unique<s_IPD::localHelperFields::TemperatureEnergyField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(temperatureEnergyField));

            // z^star IPD input field, z^star = = average(q^2) / average(q) ;for q charge number of ion, unitless,
            //  not weighted
            auto zStarField
                = std::make_unique<s_IPD::localHelperFields::ZStarField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(zStarField));

            // unit: 1/unit_length^3
            auto freeElectronDensityField
                = std::make_unique<s_IPD::localHelperFields::FreeElectronDensityField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(freeElectronDensityField));
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

        /** check for and apply a single step of the IPD-ionization cascade
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
        template<typename T_AtomicPhysicsIonSpeciesList, bool T_SkipFinishedSuperCell>
        HINLINE static void applyIPDIonization(picongpu::MappingDesc const mappingDesc, uint32_t const)
        {
            using ForEachIonSpeciesApplyIPDIonization = pmacc::meta::ForEach<
                T_AtomicPhysicsIonSpeciesList,
                s_IPD::stage::ApplyIPDIonization<
                    boost::mpl::_1,
                    StewartPyattIPD<T_TemperatureFunctor, T_useSCFLYextendedStewartPyattIPD>,
                    std::integral_constant<bool, T_SkipFinishedSuperCell>>>;
            ForEachIonSpeciesApplyIPDIonization{}(mappingDesc);
        };

        /** get the super cell constant input
         *
         * @param superCellFieldIdx index of superCell in superCellField(without guards)
         * @param debyeLengthBox deviceDataBox giving access to the local debye length for all local superCells,
         *  sim.unit.length(), not weighted
         * @param temperatureEnergyBox deviceDataBox giving access to the local temperature * k_Boltzman for all
         *  local superCells, in UNIT_ENERGY, not weighted
         * @param zStarBox deviceDataBox giving access to the local z^Star value, = average(q^2) / average(q),
         *  for all local superCells, unitless, not weighted
         */
        template<
            typename T_DebyeLengthBox,
            typename T_TemperatureEnergyBox,
            typename T_ZStarBox,
            typename T_FreeElectronDensityBox>
        HDINLINE static SuperCellConstantInput getSuperCellConstantInput(
            pmacc::DataSpace<simDim> const superCellFieldIdx,
            T_DebyeLengthBox const debyeLengthBox,
            T_TemperatureEnergyBox const temperatureEnergyBox,
            T_ZStarBox const zStarBox,
            T_FreeElectronDensityBox const freeElectronDensityBox)
        {
            // eV, not weighted
            float_X temperatureTimesk_Boltzman = temperatureEnergyBox(superCellFieldIdx);
            // unit_length, not weighted
            float_X debyeLength = debyeLengthBox(superCellFieldIdx);
            // unitless, not weighted
            float_X zStar = zStarBox(superCellFieldIdx);
            // 1/unit_length^3
            float_X freeElectronDensity = freeElectronDensityBox(superCellFieldIdx);

            return SuperCellConstantInput(temperatureTimesk_Boltzman, debyeLength, zStar, freeElectronDensity);
        }

        /** calculate ionization potential depression
         *
         * @param superCellConstantInput struct storing the values of all super cell constant IPD input parameters
         * @param chargeState charge state of the ion before ionization
         *
         * @return unit: eV, not weighted
         */
        HDINLINE static float_X ipd(SuperCellConstantInput const superCellConstantInput, uint8_t const chargeState)
        {
            // UNIT_CHARGE^2 / ( unitless * UNIT_CHARGE^2 * UNIT_TIME^2 / (UNIT_LENGTH^3 * UNIT_MASS))
            // = UNIT_MASS * UNIT_LENGTH^3 * UNIT_TIME^(-2)
            constexpr float_X constFactor
                = pmacc::math::cPow(picongpu::sim.pic.getElectronCharge(), 2u)
                  / (4._X * static_cast<float_X>(picongpu::PI) * picongpu::sim.pic.getEps0());

            // UNIT_ENERGY/eV
            constexpr float_X eV = sim.pic.get_eV();

            float_X const chargeStateAsFloat = static_cast<float_X>(chargeState);

            // UNIT_MASS * UNIT_LENGTH^3 / UNIT_TIME^2 * 1/(eV * UNIT_ENERGY/eV * UNIT_LENGTH)
            // = unitless, not weighted
            //! @details choice of core charge as (charge state + 1), according to [2]
            float_X const K
                = (pmacc::math::isApproxZero(
                      superCellConstantInput.temperatureTimesk_Boltzman * superCellConstantInput.debyeLength))
                      ? 0._X
                      : constFactor * (chargeStateAsFloat + 1._X)
                            / (superCellConstantInput.temperatureTimesk_Boltzman * eV
                               * superCellConstantInput.debyeLength);

            // eV, not weighted
            //! @details based on equation (5) from Stewart-Pyatt(1966)
            float_X const stewartPyattIPD
                = superCellConstantInput.temperatureTimesk_Boltzman
                  * (math::pow((3._X * (superCellConstantInput.zStar + 1._X) * K + 1._X), 2._X / 3._X) - 1._X)
                  / (2._X * (superCellConstantInput.zStar + 1._X));

            //! @details additional IPD contribution as implemented in SCFLY,
            float_X result = 0._X;
            if constexpr(T_useSCFLYextendedStewartPyattIPD)
            {
                /** @details as defined in [2] on p. 259 in the text before equation (4)
                 *
                 * SCFLY source code definition is at,
                 *  - base definition,
                 *      https://github.com/ComputationalRadiationPhysics/scfly/blob/5d9b0a997bb6688b04e1a34a7fa8db0f2657106a/code/src/scdrv.f#L3348-L3349
                 *  - in place modification to account for charge state,
                 *      https://github.com/ComputationalRadiationPhysics/scfly/blob/5d9b0a997bb6688b04e1a34a7fa8db0f2657106a/code/src/scdrv.f#L3536
                 */
                // (1/(1/unit_length^3))^(1/3) = unit_length
                float_X const ionSphereRadius = math::pow(
                    0.75_X * (chargeStateAsFloat + 1._X) / (picongpu::PI * superCellConstantInput.freeElectronDensity),
                    1. / 3.);

                // (eV * cm) * m/cm /(m/unit_length) = eV * m * unit_length/m = eV * unit_length
                constexpr float_X constFactorSCFLY = 1.45e-7_X * 1.e-2_X / sim.unit.length();

                // 1 * eV * unit_length / sqrt(unit_length^2 + unit_length^2) = eV, not weighted
                float_X const scfly_secondBranch
                    = constFactorSCFLY * (chargeStateAsFloat + 1._X)
                      / math::sqrt(
                          (ionSphereRadius / 1.5_X) * (ionSphereRadius / 1.5_X)
                          + (superCellConstantInput.debyeLength) * (superCellConstantInput.debyeLength));

                /** @details the factor zz = (chargeState + 1) before the minimum in the SCFLY implementation has been
                 *  moved inside the SCFLY and stewartPyattIPD branches for better encapsulation */
                result = math::min(scfly_secondBranch, stewartPyattIPD);
            }
            else
            {
                result = stewartPyattIPD;
            }

            // eV, not weighted
            return result;
        }

        /** call a kernel, appending the IPD-input to the kernel's input
         *
         * @tparam T_Kernel kernel to call
         * @tparam T_chunkSize chunk size to use in call
         * @param dc picongpu data connector
         * @param kernelInput input to pass to kernel in addition to IPD-input
         */
        template<typename T_Kernel, uint32_t T_chunkSize, typename... T_KernelInput>
        HINLINE static void callKernelWithIPDInput(
            pmacc::DataConnector& dc,
            pmacc::AreaMapping<CORE + BORDER, picongpu::MappingDesc>& mapper,
            T_KernelInput... kernelInput)
        {
            auto& debyeLengthField
                = *dc.get<s_IPD::localHelperFields::DebyeLengthField<picongpu::MappingDesc>>("DebyeLengthField");
            auto& temperatureEnergyField
                = *dc.get<s_IPD::localHelperFields::TemperatureEnergyField<picongpu::MappingDesc>>(
                    "TemperatureEnergyField");
            auto& zStarField = *dc.get<s_IPD::localHelperFields::ZStarField<picongpu::MappingDesc>>("ZStarField");
            auto& freeElectronDensityField
                = *dc.get<s_IPD::localHelperFields::FreeElectronDensityField<picongpu::MappingDesc>>(
                    "FreeElectronDensityField");

            PMACC_LOCKSTEP_KERNEL(T_Kernel())
                .template config<T_chunkSize>(mapper.getGridDim())(
                    mapper,
                    kernelInput...,
                    debyeLengthField.getDeviceDataBox(),
                    temperatureEnergyField.getDeviceDataBox(),
                    zStarField.getDeviceDataBox(),
                    freeElectronDensityField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
