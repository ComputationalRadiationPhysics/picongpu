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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
// need picongpu::simDim from picongpu/param/dimension.param
//  and picongpu/param/atomicPhysics_debug.param

#include "picongpu/particles/InitFunctors.hpp"
#include "picongpu/particles/atomicPhysics/AtomicPhysicsSuperCellFields.hpp"
#include "picongpu/particles/atomicPhysics/IPDModel.param" /// @todo move to atomicPhysics.param, Brian Marre, 2024
#include "picongpu/particles/atomicPhysics/SetTemperature.hpp"
#include "picongpu/particles/atomicPhysics/debug/stage/DumpAllIonsToConsole.hpp"
#include "picongpu/particles/atomicPhysics/debug/stage/DumpRateCacheToConsole.hpp"
#include "picongpu/particles/atomicPhysics/debug/stage/DumpSuperCellDataToConsole.hpp"
#include "picongpu/particles/atomicPhysics/stage/BinElectrons.hpp"
#include "picongpu/particles/atomicPhysics/stage/CalculateStepLength.hpp"
#include "picongpu/particles/atomicPhysics/stage/CheckForOverSubscription.hpp"
#include "picongpu/particles/atomicPhysics/stage/CheckPresence.hpp"
#include "picongpu/particles/atomicPhysics/stage/ChooseTransition.hpp"
#include "picongpu/particles/atomicPhysics/stage/ChooseTransitionType.hpp"
#include "picongpu/particles/atomicPhysics/stage/DecelerateElectrons.hpp"
#include "picongpu/particles/atomicPhysics/stage/FillLocalRateCache.hpp"
#include "picongpu/particles/atomicPhysics/stage/FixAtomicState.hpp"
#include "picongpu/particles/atomicPhysics/stage/LoadAtomicInputData.hpp"
#include "picongpu/particles/atomicPhysics/stage/RecordChanges.hpp"
#include "picongpu/particles/atomicPhysics/stage/RecordSuggestedChanges.hpp"
#include "picongpu/particles/atomicPhysics/stage/ResetAcceptedStatus.hpp"
#include "picongpu/particles/atomicPhysics/stage/ResetDeltaWeightElectronHistogram.hpp"
#include "picongpu/particles/atomicPhysics/stage/ResetLocalRateCache.hpp"
#include "picongpu/particles/atomicPhysics/stage/ResetLocalTimeStepField.hpp"
#include "picongpu/particles/atomicPhysics/stage/RollForOverSubscription.hpp"
#include "picongpu/particles/atomicPhysics/stage/SpawnIonizationElectrons.hpp"
#include "picongpu/particles/atomicPhysics/stage/UpdateTimeRemaining.hpp"

#include <pmacc/device/Reduce.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>
#include <string>

// debug only
#include "picongpu/particles/atomicPhysics/debug/TestRateCalculation.hpp"

#include <iostream>


namespace picongpu::simulation::stage
{
    namespace detail
    {
        /** atomic physics stage
         *
         * models excited atomic state and ionization dynamics
         *
         * @note one instance of this class is initialized and it's operator() called for every time step
         *
         * @tparam T_AtomicPhysicsIonSpecies list of all ion species to partake in the atomicPhysics step
         * @tparam T_OnlyIPDIonSpecies list of all ion species to be partake in the IPD calculation in addition to
         *  the atomicPhysics ion species
         * @tparam T_AtomicPhysicsElectronSpecies list of all electrons species to partake in the atomicPhysics step
         * @tparam T_OnlyIPDElectronSpecies list of all electron species to partake in the IPD calculation in addition
         *  to the atomicPhysics electron species
         * @tparam T_numberAtomicPhysicsIonSpecies specialization template parameter used to prevent compilation of all
         *  atomicPhysics kernels if no atomic physics species is present.
         */
        template<
            typename T_AtomicPhysicsIonSpecies,
            typename T_OnlyIPDIonSpecies,
            typename T_AtomicPhysicsElectronSpecies,
            typename T_OnlyIPDElectronSpecies,
            uint32_t T_numberAtomicPhysicsIonSpecies>
        struct AtomicPhysics
        {
        private:
            // linearized dataBox of SuperCellField
            template<typename T_Field>
            using S_LinearizedBox = DataBoxDim1Access<typename T_Field::DataBoxType>;

            using S_OverSubscribedField
                = picongpu::particles::atomicPhysics::localHelperFields::LocalElectronHistogramOverSubscribedField<
                    picongpu::MappingDesc>;
            using S_TimeRemainingField
                = particles::atomicPhysics::localHelperFields ::LocalTimeRemainingField<picongpu::MappingDesc>;
            using S_FoundUnboundField
                = particles::atomicPhysics::localHelperFields ::LocalFoundUnboundIonField<picongpu::MappingDesc>;

            // species Lists
            //{
            using AtomicPhysicsElectronSpecies = T_AtomicPhysicsElectronSpecies;
            using OnlyIPDElectronSpecies = T_OnlyIPDElectronSpecies;
            using AtomicPhysicsIonSpecies = T_AtomicPhysicsIonSpecies;
            using OnlyIPDIonSpecies = T_OnlyIPDIonSpecies;

            //! list of all electron species for IPD
            using IPDElectronSpecies = MakeSeq_t<AtomicPhysicsElectronSpecies, OnlyIPDElectronSpecies>;
            //! list of all ion species for IPD
            using IPDIonSpecies = MakeSeq_t<AtomicPhysicsIonSpecies, OnlyIPDIonSpecies>;
            //}

            //! set local timeRemaining to PIC-time step
            HINLINE static void setTimeRemaining()
            {
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
                auto& localTimeRemainingField = *dc.get<S_TimeRemainingField>("LocalTimeRemainingField");
                localTimeRemainingField.getDeviceBuffer().setValue(picongpu::DELTA_T); // UNIT_TIME
            }

            //! reset the histogram on device side
            HINLINE static void resetHistograms()
            {
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
                auto& localElectronHistogramField
                    = *dc.get<particles::atomicPhysics::electronDistribution::LocalHistogramField<
                        picongpu::atomicPhysics::ElectronHistogram,
                        picongpu::MappingDesc>>("Electron_localHistogramField");
                localElectronHistogramField.getDeviceBuffer().setValue(picongpu::atomicPhysics::ElectronHistogram());
            }

            //! reset localFoundUnboundIonField on device side
            HINLINE static void resetFoundUnboundIon()
            {
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
                auto& localFoundUnboundIonField = *dc.get<S_FoundUnboundField>("LocalFoundUnboundIonField");
                localFoundUnboundIonField.getDeviceBuffer().setValue(0._X);
            };

            //! print electron histogram to console, debug only
            template<bool T_printOnlyOverSubscribed>
            HINLINE static void printHistogramToConsole(picongpu::MappingDesc const mappingDesc)
            {
                picongpu::particles::atomicPhysics::stage::DumpSuperCellDataToConsole<
                    picongpu::particles::atomicPhysics::electronDistribution::
                        LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>,
                    picongpu::particles::atomicPhysics::electronDistribution::PrintHistogramToConsole<
                        T_printOnlyOverSubscribed>>{}(mappingDesc, "Electron_localHistogramField");
            }

            //! print LocalElectronHistogramOverSubscribedField to console, debug only
            HINLINE static void printOverSubscriptionFieldToConsole(picongpu::MappingDesc const mappingDesc)
            {
                picongpu::particles::atomicPhysics::stage::DumpSuperCellDataToConsole<
                    picongpu::particles::atomicPhysics::localHelperFields::LocalElectronHistogramOverSubscribedField<
                        picongpu::MappingDesc>,
                    picongpu::particles::atomicPhysics::localHelperFields::PrintOverSubcriptionFieldToConsole>{}(
                    mappingDesc,
                    "LocalElectronHistogramOverSubscribedField");
            }

            //! print rejectionProbabilityCache to console, debug only
            HINLINE static void printRejectionProbabilityCacheToConsole(picongpu::MappingDesc const mappingDesc)
            {
                picongpu::particles::atomicPhysics::stage::DumpSuperCellDataToConsole<
                    picongpu::particles::atomicPhysics::localHelperFields ::LocalRejectionProbabilityCacheField<
                        picongpu::MappingDesc>,
                    picongpu::particles::atomicPhysics::localHelperFields ::PrintRejectionProbabilityCacheToConsole<
                        true>>{}(mappingDesc, "LocalRejectionProbabilityCacheField");
            }

            //! print local time remaining to console, debug only
            HINLINE static void printTimeRemaingToConsole(picongpu::MappingDesc const mappingDesc)
            {
                picongpu::particles::atomicPhysics::stage::DumpSuperCellDataToConsole<
                    picongpu::particles::atomicPhysics::localHelperFields::LocalTimeRemainingField<
                        picongpu::MappingDesc>,
                    picongpu::particles::atomicPhysics::localHelperFields::PrintTimeRemaingToConsole>{}(
                    mappingDesc,
                    "LocalTimeRemainingField");
            }

            //! print local time step to console, debug only
            HINLINE static void printTimeStepToConsole(picongpu::MappingDesc const mappingDesc)
            {
                picongpu::particles::atomicPhysics::stage::DumpSuperCellDataToConsole<
                    picongpu::particles::atomicPhysics::localHelperFields::LocalTimeStepField<picongpu::MappingDesc>,
                    picongpu::particles::atomicPhysics::localHelperFields::PrintTimeStepToConsole>{}(
                    mappingDesc,
                    "LocalTimeStepField");
            }

        public:
            AtomicPhysics() = default;

            //! atomic physics stage sub-stage calls
            void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
            {
                // debug only
                using ForEachIonSpeciesDumpRateCacheToConsole = pmacc::meta::ForEach<
                    AtomicPhysicsIonSpecies,
                    particles::atomicPhysics::stage::DumpRateCacheToConsole<boost::mpl::_1>>;

                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

                auto& localTimeRemainingField = *dc.get<S_TimeRemainingField>("LocalTimeRemainingField");
                DataSpace<picongpu::simDim> const fieldGridLayoutTimeRemaining
                    = localTimeRemainingField.getGridLayout().sizeWithoutGuardND();

                auto& localFoundUnboundIonField = *dc.get<S_FoundUnboundField>("LocalFoundUnboundIonField");
                DataSpace<picongpu::simDim> const fieldGridLayoutFoundUnbound
                    = localFoundUnboundIonField.getGridLayout().sizeWithoutGuardND();

                auto& localElectronHistogramOverSubscribedField
                    = *dc.get<S_OverSubscribedField>("LocalElectronHistogramOverSubscribedField");
                DataSpace<picongpu::simDim> const fieldGridLayoutOverSubscription
                    = localElectronHistogramOverSubscribedField.getGridLayout().sizeWithoutGuardND();

                /// @todo find better way than hard code old value, Brian Marre, 2023
                // `static` avoids that reduce is allocating each time step memory, which will reduce the performance.
                static pmacc::device::Reduce deviceLocalReduce = pmacc::device::Reduce(static_cast<uint32_t>(1200u));

                setTimeRemaining(); // = (Delta t)_PIC
                // fix atomic state and charge state inconsistency
                using ForEachIonSpeciesFixAtomicState = pmacc::meta::
                    ForEach<AtomicPhysicsIonSpecies, particles::atomicPhysics::stage::FixAtomicState<boost::mpl::_1>>;
                ForEachIonSpeciesFixAtomicState{}(mappingDesc);

                // atomicPhysics sub-stepping loop
                bool timeRemainingLessThanZero = false;
                do
                {
                    // particle[accepted_] = false, in each macro ion
                    using ForEachIonSpeciesResetAcceptedStatus = pmacc::meta::ForEach<
                        AtomicPhysicsIonSpecies,
                        particles::atomicPhysics::stage::ResetAcceptedStatus<boost::mpl::_1>>;
                    ForEachIonSpeciesResetAcceptedStatus{}(mappingDesc);

                    resetHistograms();

                    if constexpr(picongpu::atomicPhysics::debug::scFlyComparison::FORCE_CONSTANT_ELECTRON_TEMPERATURE)
                    {
                        using ForEachElectronSpeciesSetTemperature = pmacc::meta::ForEach<
                            AtomicPhysicsElectronSpecies,
                            picongpu::particles::
                                Manipulate<picongpu::particles::atomicPhysics::SetTemperature, boost::mpl::_1>>;
                        ForEachElectronSpeciesSetTemperature{}(currentStep);
                    }

                    using ForEachElectronSpeciesBinElectrons = pmacc::meta::ForEach<
                        AtomicPhysicsElectronSpecies,
                        particles::atomicPhysics::stage::BinElectrons<boost::mpl::_1>>;
                    ForEachElectronSpeciesBinElectrons{}(mappingDesc);

                    // calculate ionization potential depression parameters for every superCell
                    picongpu::atomicPhysics::IPDModel::
                        template calculateIPDInput<T_numberAtomicPhysicsIonSpecies, IPDIonSpecies, IPDElectronSpecies>(
                            mappingDesc,
                            currentStep);

                    if constexpr(picongpu::atomicPhysics::debug::electronHistogram::PRINT_TO_CONSOLE)
                    {
                        printHistogramToConsole</*print all bins*/ false>(mappingDesc);
                    }

                    // timeStep = localTimeRemaining
                    picongpu::particles::atomicPhysics::stage::ResetLocalTimeStepField<
                        T_numberAtomicPhysicsIonSpecies>()(mappingDesc);

                    using ForEachIonSpeciesResetLocalRateCache = pmacc::meta::ForEach<
                        AtomicPhysicsIonSpecies,
                        particles::atomicPhysics::stage::ResetLocalRateCache<boost::mpl::_1>>;
                    ForEachIonSpeciesResetLocalRateCache{}();

                    // check which atomic states are actually present in each superCell
                    using ForEachIonSpeciesCheckPresenceOfAtomicStates = pmacc::meta::ForEach<
                        AtomicPhysicsIonSpecies,
                        particles::atomicPhysics::stage::CheckPresence<boost::mpl::_1>>;
                    ForEachIonSpeciesCheckPresenceOfAtomicStates{}(mappingDesc);

                    // fill rate cache with diagonal elements of rate matrix
                    using ForEachIonSpeciesFillLocalRateCache = pmacc::meta::ForEach<
                        AtomicPhysicsIonSpecies,
                        particles::atomicPhysics::stage::FillLocalRateCache<boost::mpl::_1>>;
                    ForEachIonSpeciesFillLocalRateCache{}(mappingDesc);

                    if constexpr(picongpu::atomicPhysics::debug::rateCache::PRINT_TO_CONSOLE)
                        ForEachIonSpeciesDumpRateCacheToConsole{}(mappingDesc);

                    // min(1/(-R_ii)) * alpha, calculate local atomicPhysics time step length
                    using ForEachIonSpeciesCalculateStepLength = pmacc::meta::ForEach<
                        AtomicPhysicsIonSpecies,
                        particles::atomicPhysics::stage::CalculateStepLength<boost::mpl::_1>>;
                    ForEachIonSpeciesCalculateStepLength{}(mappingDesc);

                    // choose transition loop
                    bool histogramOverSubscribed = true;
                    while(histogramOverSubscribed)
                    {
                        // randomly roll transition for each not yet accepted macro ion
                        using ForEachIonSpeciesChooseTransitionType = pmacc::meta::ForEach<
                            AtomicPhysicsIonSpecies,
                            particles::atomicPhysics::stage::ChooseTransitionType<boost::mpl::_1>>;
                        ForEachIonSpeciesChooseTransitionType{}(mappingDesc, currentStep);

                        using ForEachIonSpeciesChooseTransition = pmacc::meta::ForEach<
                            AtomicPhysicsIonSpecies,
                            particles::atomicPhysics::stage::ChooseTransition<boost::mpl::_1>>;
                        ForEachIonSpeciesChooseTransition{}(mappingDesc, currentStep);

                        picongpu::particles::atomicPhysics::stage::ResetDeltaWeightElectronHistogram<
                            T_numberAtomicPhysicsIonSpecies>{}(mappingDesc);

                        // record all shared resources usage by accepted transitions
                        using ForEachIonSpeciesRecordSuggestedChanges = pmacc::meta::ForEach<
                            AtomicPhysicsIonSpecies,
                            particles::atomicPhysics::stage::RecordSuggestedChanges<boost::mpl::_1>>;
                        ForEachIonSpeciesRecordSuggestedChanges{}(mappingDesc);

                        // check bins for over subscription --> localElectronHistogramOverSubscribedField
                        picongpu::particles::atomicPhysics::stage::CheckForOverSubscription<
                            T_numberAtomicPhysicsIonSpecies>{}(mappingDesc);

                        auto linearizedOverSubscribedBox = S_LinearizedBox<S_OverSubscribedField>(
                            localElectronHistogramOverSubscribedField.getDeviceDataBox(),
                            fieldGridLayoutOverSubscription);

                        // debug only
                        if constexpr(picongpu::atomicPhysics::debug::rejectionProbabilityCache::PRINT_TO_CONSOLE)
                        {
                            std::cout << "\t\t a histogram oversubscribed?: "
                                      << ((static_cast<bool>(deviceLocalReduce(
                                              pmacc::math::operation::Or(),
                                              linearizedOverSubscribedBox,
                                              fieldGridLayoutOverSubscription.productOfComponents())))
                                              ? "true"
                                              : "false")
                                      << std::endl;

                            printOverSubscriptionFieldToConsole(mappingDesc);
                            printRejectionProbabilityCacheToConsole(mappingDesc);
                            printHistogramToConsole</*print only oversubscribed*/ true>(mappingDesc);
                        }

                        // check whether a least one histogram is oversubscribed
                        if(!static_cast<bool>(deviceLocalReduce(
                               pmacc::math::operation::Or(),
                               linearizedOverSubscribedBox,
                               fieldGridLayoutOverSubscription.productOfComponents())))
                        {
                            histogramOverSubscribed = false;
                        }
                        // at least one superCell electron histogram over subscribed

                        if(histogramOverSubscribed)
                        {
                            // remove overSubscription
                            bool noHistogramOverSubscribed = false;
                            do
                            {
                                // debug only
                                if constexpr(picongpu::atomicPhysics::debug::kernel::rollForOverSubscription::
                                                 PRINT_DEBUG_TO_CONSOLE)
                                {
                                    printOverSubscriptionFieldToConsole(mappingDesc);
                                    printHistogramToConsole</*print only oversubscribed*/ true>(mappingDesc);

                                    if constexpr(picongpu::atomicPhysics::debug::rejectionProbabilityCache::
                                                     PRINT_TO_CONSOLE)
                                        printRejectionProbabilityCacheToConsole(mappingDesc);
                                }

                                using ForEachIonSpeciesRollForOverSubscription = pmacc::meta::ForEach<
                                    AtomicPhysicsIonSpecies,
                                    particles::atomicPhysics::stage::RollForOverSubscription<boost::mpl::_1>>;
                                ForEachIonSpeciesRollForOverSubscription{}(mappingDesc, currentStep);

                                picongpu::particles::atomicPhysics::stage::ResetDeltaWeightElectronHistogram<
                                    T_numberAtomicPhysicsIonSpecies>{}(mappingDesc);

                                // record all shared resources usage by accepted transitions
                                ForEachIonSpeciesRecordSuggestedChanges{}(mappingDesc);

                                // check bins for over subscription --> localElectronHistogramOverSubscribedField
                                picongpu::particles::atomicPhysics::stage::CheckForOverSubscription<
                                    T_numberAtomicPhysicsIonSpecies>()(mappingDesc);

                                auto linearizedOverSubscribedBox = S_LinearizedBox<S_OverSubscribedField>(
                                    localElectronHistogramOverSubscribedField.getDeviceDataBox(),
                                    fieldGridLayoutOverSubscription);

                                noHistogramOverSubscribed = !static_cast<bool>(deviceLocalReduce(
                                    pmacc::math::operation::Or(),
                                    linearizedOverSubscribedBox,
                                    fieldGridLayoutOverSubscription.productOfComponents()));
                            } // end remove over subscription loop
                            while(noHistogramOverSubscribed);
                        }
                    } // end choose transition loop

                    if constexpr(picongpu::atomicPhysics::debug::timeRemaining::PRINT_TO_CONSOLE)
                        printTimeRemaingToConsole(mappingDesc);
                    if constexpr(picongpu::atomicPhysics::debug::timeStep::PRINT_TO_CONSOLE)
                        printTimeStepToConsole(mappingDesc);

                    /** update atomic state and accumulate delta energy for delta energy histogram
                     *
                     * @note may already update the atomic state since the following kernels DecelerateElectrons and
                     * SpawnIonizationElectrons only use the transitionIndex particle attribute */
                    using ForEachIonSpeciesRecordChanges = pmacc::meta::ForEach<
                        AtomicPhysicsIonSpecies,
                        particles::atomicPhysics::stage::RecordChanges<boost::mpl::_1>>;
                    ForEachIonSpeciesRecordChanges{}(mappingDesc);

                    /** @note DecelerateElectrons must be called before SpawnIonizationElectrons such that we only
                     * change electrons that actually contributed to the histogram*/
                    using ForEachElectronSpeciesDecelerateElectrons = pmacc::meta::ForEach<
                        AtomicPhysicsElectronSpecies,
                        particles::atomicPhysics::stage::DecelerateElectrons<boost::mpl::_1>>;
                    ForEachElectronSpeciesDecelerateElectrons{}(mappingDesc);

                    using ForEachIonSpeciesSpawnIonizationElectrons = pmacc::meta::ForEach<
                        AtomicPhysicsIonSpecies,
                        particles::atomicPhysics::stage::SpawnIonizationElectrons<boost::mpl::_1>>;
                    ForEachIonSpeciesSpawnIonizationElectrons{}(mappingDesc, currentStep);

                    // pressure ionization loop, ends when no ion in unbound state anymore
                    bool foundUnbound = true;
                    do
                    {
                        resetFoundUnboundIon();
                        picongpu::atomicPhysics::IPDModel::template calculateIPDInput<
                            T_numberAtomicPhysicsIonSpecies,
                            IPDIonSpecies,
                            IPDElectronSpecies>(mappingDesc, currentStep);
                        picongpu::atomicPhysics::IPDModel::template applyPressureIonization<AtomicPhysicsIonSpecies>(
                            mappingDesc,
                            currentStep);

                        auto linearizedFoundUnboundIonBox = S_LinearizedBox<S_FoundUnboundField>(
                            localFoundUnboundIonField.getDeviceDataBox(),
                            fieldGridLayoutFoundUnbound);

                        foundUnbound = static_cast<bool>(deviceLocalReduce(
                            pmacc::math::operation::Or(),
                            linearizedFoundUnboundIonBox,
                            fieldGridLayoutFoundUnbound.productOfComponents()));
                    } // end pressure ionization loop
                    while(foundUnbound);

                    // timeRemaining -= timeStep
                    picongpu::particles::atomicPhysics::stage::UpdateTimeRemaining<T_numberAtomicPhysicsIonSpecies>()(
                        mappingDesc);

                    auto linearizedTimeRemainingBox = S_LinearizedBox<S_TimeRemainingField>(
                        localTimeRemainingField.getDeviceDataBox(),
                        fieldGridLayoutTimeRemaining);

                    timeRemainingLessThanZero = deviceLocalReduce(
                                                    pmacc::math::operation::Max(),
                                                    linearizedTimeRemainingBox,
                                                    fieldGridLayoutTimeRemaining.productOfComponents())
                        <= 0._X;
                } // end atomicPhysics sub-stepping loop
                while(!timeRemainingLessThanZero);
            }
        };

        //! dummy version for no atomic physics ion species in input
        template<
            typename T_AtomicPhysicsIonSpecies,
            typename T_OnlyIPDIonSpecies,
            typename T_AtomicPhysicsElectronSpecies,
            typename T_OnlyIPDElectronSpecies>
        struct AtomicPhysics<
            T_AtomicPhysicsIonSpecies,
            T_OnlyIPDIonSpecies,
            T_AtomicPhysicsElectronSpecies,
            T_OnlyIPDElectronSpecies,
            0u>
        {
            void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
            {
            }
        };
    } // namespace detail

    /** public interface of AtomicPhysics stage
     *
     * @note indirection necessary to avoid always compiling atomicPhysics stages
     */
    struct AtomicPhysics
    {
        using SpeciesRepresentingAtomicPhysicsIons =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsIon<>>::type;
        auto static constexpr numberAtomicPhysicsSpecies = pmacc::mp_size<
            pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsIon<>>::type>::value;
        static bool constexpr atomicPhysicsActive = (numberAtomicPhysicsSpecies >= 1);

    private:
        /** load the atomic input files for each species with atomicData
         *
         * create an atomicData data base object for each atomicPhysics ion species and stores them in the data
         * connector
         *
         * @todo allow reuse of atomicData dataBase objects in between species, Brian Marre, 2022
         */
        void loadAtomicInputData(DataConnector& dataConnector)
        {
            using SpeciesWithAtomicPhysicsInputData =
                typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, atomicDataType<>>::type;

            pmacc::meta::ForEach<
                SpeciesWithAtomicPhysicsInputData,
                particles::atomicPhysics::stage::LoadAtomicInputData<boost::mpl::_1>>
                ForEachIonSpeciesLoadAtomicInputData;
            ForEachIonSpeciesLoadAtomicInputData(dataConnector);
        }

    public:
        //! @details indirection necessary to prevent compiling atomicPhysics kernels if no atomicPhysics species exist
        AtomicPhysics(picongpu::MappingDesc const mappingDesc)
        {
            // init atomicPhysics fields and buffers
            if constexpr(atomicPhysicsActive)
            {
                DataConnector& dc = Environment<>::get().DataConnector();

                loadAtomicInputData(dc);
                picongpu::particles::atomicPhysics::AtomicPhysicsSuperCellFields::create(dc, mappingDesc);
                picongpu::atomicPhysics::IPDModel::createHelperFields(dc, mappingDesc);
            }

            if constexpr(picongpu::atomicPhysics::debug::rateCalculation::RUN_UNIT_TESTS)
            {
                auto test = particles::atomicPhysics::debug::TestRateCalculation<10u>();
                std::cout << "TestRateCalculation:" << std::endl;
                test.testAll();
            }
        }

        void fixAtomicStateInit(picongpu::MappingDesc const mappingDesc)
        {
            using ForEachIonSpeciesFixAtomicState = pmacc::meta::ForEach<
                SpeciesRepresentingAtomicPhysicsIons,
                particles::atomicPhysics::stage::FixAtomicState<boost::mpl::_1>>;
            ForEachIonSpeciesFixAtomicState{}(mappingDesc);
        }

        void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
        {
            if constexpr(atomicPhysicsActive)
            {
                /** list of all species of macro particles that partake in atomicPhysics as ions
                 *
                 * @attention atomicPhysics  assumes that all species with isAtomicPhysicsIon flag also have the
                 * required atomic data flags, see picongpu/param/speciesAttributes.param for details
                 */
                using AtomicPhysicsIonSpecies =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsIon<>>::type;
                //! list of all only IPD partaking ion species
                using OnlyIPDIonSpecies =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isOnlyIPDIon<>>::type;

                //! list of all species of macro particles that partake in atomicPhysics as electrons
                using AtomicPhysicsElectronSpecies =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsElectron<>>::type;
                //! list of all only IPD partaking electron species
                using OnlyIPDElectronSpecies =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isOnlyIPDElectron<>>::type;

                detail::AtomicPhysics<
                    AtomicPhysicsIonSpecies,
                    OnlyIPDIonSpecies,
                    AtomicPhysicsElectronSpecies,
                    OnlyIPDElectronSpecies,
                    numberAtomicPhysicsSpecies>{}(mappingDesc, currentStep);
            }
        }
    };
} // namespace picongpu::simulation::stage
