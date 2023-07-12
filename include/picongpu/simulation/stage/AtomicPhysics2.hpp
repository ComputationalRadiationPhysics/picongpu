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
//  and picongpu/param/atomicPhysics2_debug.param

#include "picongpu/particles/InitFunctors.hpp"
#include "picongpu/particles/atomicPhysics2/AtomicPhysicsSuperCellFields.hpp"
#include "picongpu/particles/atomicPhysics2/IPDModel.param"
#include "picongpu/particles/atomicPhysics2/SetTemperature.hpp"
#include "picongpu/particles/atomicPhysics2/stage/BinElectrons.hpp"
#include "picongpu/particles/atomicPhysics2/stage/CalculateStepLength.hpp"
#include "picongpu/particles/atomicPhysics2/stage/CheckForOverSubscription.hpp"
#include "picongpu/particles/atomicPhysics2/stage/CheckPresence.hpp"
#include "picongpu/particles/atomicPhysics2/stage/ChooseTransition.hpp"
#include "picongpu/particles/atomicPhysics2/stage/ChooseTransitionType.hpp"
#include "picongpu/particles/atomicPhysics2/stage/DecelerateElectrons.hpp"
#include "picongpu/particles/atomicPhysics2/stage/DumpAllIonsToConsole.hpp"
#include "picongpu/particles/atomicPhysics2/stage/DumpRateCacheToConsole.hpp"
#include "picongpu/particles/atomicPhysics2/stage/DumpSuperCellDataToConsole.hpp"
#include "picongpu/particles/atomicPhysics2/stage/FillLocalRateCache.hpp"
#include "picongpu/particles/atomicPhysics2/stage/FixAtomicState.hpp"
#include "picongpu/particles/atomicPhysics2/stage/RecordChanges.hpp"
#include "picongpu/particles/atomicPhysics2/stage/RecordSuggestedChanges.hpp"
#include "picongpu/particles/atomicPhysics2/stage/ResetAcceptedStatus.hpp"
#include "picongpu/particles/atomicPhysics2/stage/ResetDeltaWeightElectronHistogram.hpp"
#include "picongpu/particles/atomicPhysics2/stage/ResetLocalRateCache.hpp"
#include "picongpu/particles/atomicPhysics2/stage/ResetLocalTimeStepField.hpp"
#include "picongpu/particles/atomicPhysics2/stage/RollForOverSubscription.hpp"
#include "picongpu/particles/atomicPhysics2/stage/SpawnIonizationElectrons.hpp"
#include "picongpu/particles/atomicPhysics2/stage/UpdateTimeRemaining.hpp"

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
#include <iostream>

namespace picongpu::simulation::stage
{
    /** atomic physics stage
     *
     * models excited atomic state and ionization dynamics
     *
     * @note one instance of this class is initialized and it's operator() called for every time step
     */
    struct AtomicPhysics2
    {
    private:
        // linearized dataBox of SuperCellField
        template<typename T_Field>
        using S_LinearizedBox = DataBoxDim1Access<typename T_Field::DataBoxType>;

        using S_OverSubscribedField
            = picongpu::particles::atomicPhysics2::localHelperFields::LocalElectronHistogramOverSubscribedField<
                picongpu::MappingDesc>;
        using S_TimeRemainingField
            = particles::atomicPhysics2::localHelperFields ::LocalTimeRemainingField<picongpu::MappingDesc>;
        using S_FoundUnboundField
            = particles::atomicPhysics2::localHelperFields ::LocalFoundUnboundIonField<picongpu::MappingDesc>;

        // species Lists
        //{
        //! list of all species of macro particles that partake in atomicPhysics as electrons
        using AtomicPhysicsElectronSpecies =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsElectron<>>::type;
        //! list of all only IPD partaking electron species
        using OnlyIPDElectronSpecies =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isOnlyIPDElectron<>>::type;

        /** list of all species of macro particles that partake in atomicPhysics as ions
         *
         * @attention atomicPhysics  assumes that all species with isAtomicPhysicsIon flag also have the required
         * atomic data flags, see picongpu/param/speciesAttributes.param for details
         */
        using AtomicPhysicsIonSpecies =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsIon<>>::type;
        //! list of all only IPD partaking ion species
        using OnlyIPDIonSpecies =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isOnlyIPDIon<>>::type;

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
                = *dc.get<particles::atomicPhysics2::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics2::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            localElectronHistogramField.getDeviceBuffer().setValue(picongpu::atomicPhysics2::ElectronHistogram());
        }

        //! reset localFoundUnboundIonField on device side
        HINLINE static void resetFoundUnboundIon()
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            // reset foundUnbound Field
            auto& localFoundUnboundIonField = *dc.get<S_FoundUnboundField>("LocalFoundUnboundIonField");

            localFoundUnboundIonField.getDeviceBuffer().setValue(0._X);
        };

        //! print electron histogram to console, debug only
        template<bool T_printOnlyOverSubscribed>
        HINLINE static void printHistogramToConsole(picongpu::MappingDesc const mappingDesc)
        {
            picongpu::particles::atomicPhysics2::stage::DumpSuperCellDataToConsole<
                picongpu::particles::atomicPhysics2::electronDistribution::
                    LocalHistogramField<picongpu::atomicPhysics2::ElectronHistogram, picongpu::MappingDesc>,
                picongpu::particles::atomicPhysics2::electronDistribution::PrintHistogramToConsole<
                    T_printOnlyOverSubscribed>>{}(mappingDesc, "Electron_localHistogramField");
        }

        //! print LocalElectronHistogramOverSubscribedField to console, debug only
        HINLINE static void printOverSubscriptionFieldToConsole(picongpu::MappingDesc const mappingDesc)
        {
            picongpu::particles::atomicPhysics2::stage::DumpSuperCellDataToConsole<
                picongpu::particles::atomicPhysics2::localHelperFields::LocalElectronHistogramOverSubscribedField<
                    picongpu::MappingDesc>,
                picongpu::particles::atomicPhysics2::localHelperFields::PrintOverSubcriptionFieldToConsole>{}(
                mappingDesc,
                "LocalElectronHistogramOverSubscribedField");
        }

        //! print rejectionProbabilityCache to console, debug only
        HINLINE static void printRejectionProbabilityCacheToConsole(picongpu::MappingDesc const mappingDesc)
        {
            picongpu::particles::atomicPhysics2::stage::DumpSuperCellDataToConsole<
                picongpu::particles::atomicPhysics2::localHelperFields ::LocalRejectionProbabilityCacheField<
                    picongpu::MappingDesc>,
                picongpu::particles::atomicPhysics2::localHelperFields ::PrintRejectionProbabilityCacheToConsole<
                    true>>{}(mappingDesc, "LocalRejectionProbabilityCacheField");
        }

        //! print local time remaining to console, debug only
        HINLINE static void printTimeRemaingToConsole(picongpu::MappingDesc const mappingDesc)
        {
            picongpu::particles::atomicPhysics2::stage::DumpSuperCellDataToConsole<
                picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>,
                picongpu::particles::atomicPhysics2::localHelperFields::PrintTimeRemaingToConsole>{}(
                mappingDesc,
                "LocalTimeRemainingField");
        }

        //! print local time step to console, debug only
        HINLINE static void printTimeStepToConsole(picongpu::MappingDesc const mappingDesc)
        {
            picongpu::particles::atomicPhysics2::stage::DumpSuperCellDataToConsole<
                picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeStepField<picongpu::MappingDesc>,
                picongpu::particles::atomicPhysics2::localHelperFields::PrintTimeStepToConsole>{}(
                mappingDesc,
                "LocalTimeStepField");
        }

    public:
        AtomicPhysics2() = default;

        //! atomic physics stage sub-stage calls
        void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
        {
            //! fix mismatches between boundElectrons and atomicStateCollectionIndex attributes
            using ForEachIonSpeciesFixAtomicState = pmacc::meta::
                ForEach<AtomicPhysicsIonSpecies, particles::atomicPhysics2::stage::FixAtomicState<boost::mpl::_1>>;
            //! reset macro particle attribute accepted to false for each ion species
            using ForEachIonSpeciesResetAcceptedStatus = pmacc::meta::ForEach<
                AtomicPhysicsIonSpecies,
                particles::atomicPhysics2::stage::ResetAcceptedStatus<boost::mpl::_1>>;
            //! bin electrons sub stage call for each electron species
            using ForEachElectronSpeciesBinElectrons = pmacc::meta::
                ForEach<AtomicPhysicsElectronSpecies, particles::atomicPhysics2::stage::BinElectrons<boost::mpl::_1>>;
            //! reset localRateCacheField sub stage for each ion species
            using ForEachIonSpeciesResetLocalRateCache = pmacc::meta::ForEach<
                AtomicPhysicsIonSpecies,
                particles::atomicPhysics2::stage::ResetLocalRateCache<boost::mpl::_1>>;
            //! fill rate cache with diagonal elements of rate matrix
            using ForEachIonSpeciesFillLocalRateCache = pmacc::meta::
                ForEach<AtomicPhysicsIonSpecies, particles::atomicPhysics2::stage::FillLocalRateCache<boost::mpl::_1>>;
            //! check which atomic states are actually present in each superCell
            using ForEachIonSpeciesCheckPresenceOfAtomicStates = pmacc::meta::
                ForEach<AtomicPhysicsIonSpecies, particles::atomicPhysics2::stage::CheckPresence<boost::mpl::_1>>;
            //! calculate local atomicPhysics time step length
            using ForEachIonSpeciesCalculateStepLength = pmacc::meta::ForEach<
                AtomicPhysicsIonSpecies,
                particles::atomicPhysics2::stage::CalculateStepLength<boost::mpl::_1>>;
            //! chooseTransitionType for every macro-ion
            using ForEachIonSpeciesChooseTransitionType = pmacc::meta::ForEach<
                AtomicPhysicsIonSpecies,
                particles::atomicPhysics2::stage::ChooseTransitionType<boost::mpl::_1>>;
            //! chooseTransitionType for every macro-ion
            using ForEachIonSpeciesChooseTransition = pmacc::meta::
                ForEach<AtomicPhysicsIonSpecies, particles::atomicPhysics2::stage::ChooseTransition<boost::mpl::_1>>;
            //! record suggested changes
            using ForEachIonSpeciesRecordSuggestedChanges = pmacc::meta::ForEach<
                AtomicPhysicsIonSpecies,
                particles::atomicPhysics2::stage::RecordSuggestedChanges<boost::mpl::_1>>;
            //! roll for rejection of transitions due to over subscription
            using ForEachIonSpeciesRollForOverSubscription = pmacc::meta::ForEach<
                AtomicPhysicsIonSpecies,
                particles::atomicPhysics2::stage::RollForOverSubscription<boost::mpl::_1>>;
            //! record delta energy for all transitions
            using ForEachIonSpeciesRecordChanges = pmacc::meta::
                ForEach<AtomicPhysicsIonSpecies, particles::atomicPhysics2::stage::RecordChanges<boost::mpl::_1>>;
            //! decelerate all electrons according to their bin delta energy
            using ForEachElectronSpeciesDecelerateElectrons = pmacc::meta::ForEach<
                AtomicPhysicsElectronSpecies,
                particles::atomicPhysics2::stage::DecelerateElectrons<boost::mpl::_1>>;
            //! spawn ionization created macro electrons due to atomicPhysics processes
            using ForEachIonSpeciesSpawnIonizationElectrons = pmacc::meta::ForEach<
                AtomicPhysicsIonSpecies,
                particles::atomicPhysics2::stage::SpawnIonizationElectrons<boost::mpl::_1>>;

            // debug only
            using ForEachIonSpeciesDumpToConsole = pmacc::meta::ForEach<
                AtomicPhysicsIonSpecies,
                particles::atomicPhysics2::stage::DumpAllIonsToConsole<boost::mpl::_1>>;
            using ForEachElectronSpeciesSetTemperature = pmacc::meta::ForEach<
                AtomicPhysicsElectronSpecies,
                picongpu::particles::Manipulate<picongpu::particles::atomicPhysics2::SetTemperature, boost::mpl::_1>>;
            using ForEachIonSpeciesDumpRateCacheToConsole = pmacc::meta::ForEach<
                AtomicPhysicsIonSpecies,
                particles::atomicPhysics2::stage::DumpRateCacheToConsole<boost::mpl::_1>>;

            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            // TimeRemainingSuperCellField
            auto& localTimeRemainingField = *dc.get<S_TimeRemainingField>("LocalTimeRemainingField");
            DataSpace<picongpu::simDim> const fieldGridLayoutTimeRemaining
                = localTimeRemainingField.getGridLayout().getDataSpaceWithoutGuarding();

            // FoundUnboundIonSuperCellField
            auto& localFoundUnboundIonField = *dc.get<S_FoundUnboundField>("LocalFoundUnboundIonField");
            DataSpace<picongpu::simDim> const fieldGridLayoutFoundUnbound
                = localFoundUnboundIonField.getGridLayout().getDataSpaceWithoutGuarding();

            // ElectronHistogramOverSubscribedSuperCellField
            auto& localElectronHistogramOverSubscribedField
                = *dc.get<S_OverSubscribedField>("LocalElectronHistogramOverSubscribedField");
            DataSpace<picongpu::simDim> const fieldGridLayoutOverSubscription
                = localElectronHistogramOverSubscribedField.getGridLayout().getDataSpaceWithoutGuarding();

            /// @todo find better way than hard code old value, Brian Marre, 2023
            // `static` avoids that reduce is allocating each time step memory, which will reduce the performance.
            static pmacc::device::Reduce deviceLocalReduce = pmacc::device::Reduce(static_cast<uint32_t>(1200u));

            setTimeRemaining(); // = (Delta t)_PIC
            // fix atomic state and charge state inconsistency
            ForEachIonSpeciesFixAtomicState{}(mappingDesc);

            uint16_t counterSubStep = 0u;
            // atomicPhysics sub-stepping loop, ends when timeRemaining<=0._X
            while(true)
            {
                // debug only
                std::cout << "atomicPhysics Sub-Step: " << counterSubStep << std::endl;

                // particle[accepted_] = false, in each macro ion
                ForEachIonSpeciesResetAcceptedStatus{}(mappingDesc);
                resetHistograms();

                if constexpr(picongpu::atomicPhysics2::debug::scFlyComparison::FORCE_CONSTANT_ELECTRON_TEMPERATURE)
                {
                    ForEachElectronSpeciesSetTemperature{}(currentStep);
                }

                ForEachElectronSpeciesBinElectrons{}(mappingDesc);

                // calculate ionization potential depression parameters for every superCell
                picongpu::atomicPhysics2::IPDModel::template calculateIPDInput<IPDIonSpecies, IPDElectronSpecies>(
                    mappingDesc,
                    currentStep);

                if constexpr(picongpu::atomicPhysics2::debug::electronHistogram::PRINT_TO_CONSOLE)
                {
                    printHistogramToConsole</*print all bins*/ false>(mappingDesc);
                }

                // timeStep = localTimeRemaining
                picongpu::particles::atomicPhysics2::stage::ResetLocalTimeStepField()(mappingDesc);
                ForEachIonSpeciesResetLocalRateCache{}();
                ForEachIonSpeciesCheckPresenceOfAtomicStates{}(mappingDesc);
                // R_ii = -(sum of rates of all transitions from state i to some other state j)
                ForEachIonSpeciesFillLocalRateCache{}(mappingDesc);

                if constexpr(picongpu::atomicPhysics2::debug::rateCache::PRINT_TO_CONSOLE)
                    ForEachIonSpeciesDumpRateCacheToConsole{}(mappingDesc);

                // min(1/(-R_ii)) * alpha
                ForEachIonSpeciesCalculateStepLength{}(mappingDesc);

                // debug only
                uint32_t counterOverSubscription = 0u;

                // reject overSubscription loop, ends when no histogram bin oversubscribed
                while(true)
                {
                    // randomly roll transition for each not yet accepted macro ion
                    ForEachIonSpeciesChooseTransitionType{}(mappingDesc, currentStep);
                    ForEachIonSpeciesChooseTransition{}(mappingDesc, currentStep);

                    if constexpr(picongpu::atomicPhysics2::debug::kernel::chooseTransition::
                                     DUMP_ION_DATA_TO_CONSOLE_EACH_TRY)
                    {
                        std::cout << "choose Transition loop try:" << std::endl;
                        ForEachIonSpeciesDumpToConsole{}(mappingDesc);
                    }

                    picongpu::particles::atomicPhysics2::stage::ResetDeltaWeightElectronHistogram{}(mappingDesc);
                    // record all shared resources usage by accepted transitions
                    ForEachIonSpeciesRecordSuggestedChanges{}(mappingDesc);
                    // check bins for over subscription --> localElectronHistogramOverSubscribedField
                    picongpu::particles::atomicPhysics2::stage::CheckForOverSubscription{}(mappingDesc);

                    auto linearizedOverSubscribedBox = S_LinearizedBox<S_OverSubscribedField>(
                        localElectronHistogramOverSubscribedField.getDeviceDataBox(),
                        fieldGridLayoutOverSubscription);

                    // debug only
                    if constexpr(picongpu::atomicPhysics2::debug::rejectionProbabilityCache::PRINT_TO_CONSOLE)
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

                    if(!static_cast<bool>(deviceLocalReduce(
                           pmacc::math::operation::Or(),
                           linearizedOverSubscribedBox,
                           fieldGridLayoutOverSubscription.productOfComponents())))
                    {
                        /* no superCell electron histogram marked as over subscribed in
                         *  localElectronHistogramOverSubscribedField */
                        break;
                    }
                    // at least one superCell electron histogram over subscribed

                    // remove overSubscription loop, ends when overSubscription ended by rejecting enough transitions
                    while(true)
                    {
                        // debug only
                        if constexpr(picongpu::atomicPhysics2::debug::kernel::rollForOverSubscription::
                                         PRINT_DEBUG_TO_CONSOLE)
                        {
                            if constexpr(picongpu::atomicPhysics2::debug::rejectionProbabilityCache::PRINT_TO_CONSOLE)
                            {
                                std::cout << "\t\t [" << counterOverSubscription << "] a histogram oversubscribed?: "
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
                        }

                        ForEachIonSpeciesRollForOverSubscription{}(mappingDesc, currentStep);
                        picongpu::particles::atomicPhysics2::stage::ResetDeltaWeightElectronHistogram{}(mappingDesc);
                        // record all shared resources usage by accepted transitions
                        ForEachIonSpeciesRecordSuggestedChanges{}(mappingDesc);
                        // check bins for over subscription --> localElectronHistogramOverSubscribedField
                        picongpu::particles::atomicPhysics2::stage::CheckForOverSubscription()(mappingDesc);

                        auto linearizedOverSubscribedBox = S_LinearizedBox<S_OverSubscribedField>(
                            localElectronHistogramOverSubscribedField.getDeviceDataBox(),
                            fieldGridLayoutOverSubscription);

                        if(!static_cast<bool>(deviceLocalReduce(
                               pmacc::math::operation::Or(),
                               linearizedOverSubscribedBox,
                               fieldGridLayoutOverSubscription.productOfComponents())))
                        {
                            /* no superCell electron histogram marked as over subscribed in
                             *  localElectronHistogramOverSubscribedField */
                            break;
                        }
                    }
                    // debug only
                    ++counterOverSubscription;
                } // end reject overSubscription loop

                // debug only
                std::cout << "\t counterOverSubscription: " << counterOverSubscription << std::endl;

                if constexpr(picongpu::atomicPhysics2::debug::kernel::chooseTransition::
                                 DUMP_ION_DATA_TO_CONSOLE_ALL_ACCEPTED)
                {
                    std::cout << "all accepted: current state:" << std::endl;
                    ForEachIonSpeciesDumpToConsole{}(mappingDesc);
                }
                if constexpr(picongpu::atomicPhysics2::debug::timeRemaining::PRINT_TO_CONSOLE)
                    printTimeRemaingToConsole(mappingDesc);
                if constexpr(picongpu::atomicPhysics2::debug::timeStep::PRINT_TO_CONSOLE)
                    printTimeStepToConsole(mappingDesc);


                /** update atomic state and accumulate delta energy for delta energy histogram
                 *
                 * @note may already update the atomic state since the following kernels DecelerateElectrons and
                 * SpawnIonizationElectrons only use the transitionIndex particle attribute */
                ForEachIonSpeciesRecordChanges{}(mappingDesc);
                /** @note DecelerateElectrons must be called before SpawnIonizationElectrons such that we only change
                 * electrons that actually contributed to the histogram*/
                ForEachElectronSpeciesDecelerateElectrons{}(mappingDesc);
                ForEachIonSpeciesSpawnIonizationElectrons{}(mappingDesc, currentStep);

                // pressure ionization loop, ends when no ion in unbound state anymore
                while(true)
                {
                    resetFoundUnboundIon();
                    picongpu::atomicPhysics2::IPDModel::template calculateIPDInput<IPDIonSpecies, IPDElectronSpecies>(
                        mappingDesc,
                        currentStep);
                    picongpu::atomicPhysics2::IPDModel::template applyPressureIonization<AtomicPhysicsIonSpecies>(
                        mappingDesc,
                        currentStep);

                    auto linearizedFoundUnboundIonBox = S_LinearizedBox<S_FoundUnboundField>(
                        localFoundUnboundIonField.getDeviceDataBox(),
                        fieldGridLayoutFoundUnbound);

                    if(!static_cast<bool>(deviceLocalReduce(
                           pmacc::math::operation::Or(),
                           linearizedFoundUnboundIonBox,
                           fieldGridLayoutFoundUnbound.productOfComponents())))
                    {
                        // no ion found in unbound state
                        break;
                    }
                }

                // timeRemaining -= timeStep
                picongpu::particles::atomicPhysics2::stage::UpdateTimeRemaining()(mappingDesc);

                auto linearizedTimeRemainingBox = S_LinearizedBox<S_TimeRemainingField>(
                    localTimeRemainingField.getDeviceDataBox(),
                    fieldGridLayoutTimeRemaining);

                // timeRemaining <= 0? in all local superCells?
                if(deviceLocalReduce(
                       pmacc::math::operation::Max(),
                       linearizedTimeRemainingBox,
                       fieldGridLayoutTimeRemaining.productOfComponents())
                   <= 0._X)
                {
                    break;
                }

                // debug only
                counterSubStep++;
            } // end atomicPhysics sub-stepping loop
        }
    };
} // namespace picongpu::simulation::stage
