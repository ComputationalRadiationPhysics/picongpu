/* Copyright 2022-2024 Brian Marre
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

#include "picongpu/simulation/stage/AtomicPhysics.hpp"

#include "picongpu/defines.hpp"
#include "picongpu/particles/Manipulate.hpp"
#include "picongpu/particles/atomicPhysics/AtomicPhysicsSuperCellFields.hpp"
#include "picongpu/particles/atomicPhysics/ParticleType.hpp"
#include "picongpu/particles/atomicPhysics/SetTemperature.hpp"
#include "picongpu/particles/atomicPhysics/debug/stage/DumpAllIonsToConsole.hpp"
#include "picongpu/particles/atomicPhysics/debug/stage/DumpRateCacheToConsole.hpp"
#include "picongpu/particles/atomicPhysics/debug/stage/DumpSuperCellDataToConsole.hpp"
#include "picongpu/particles/atomicPhysics/param.hpp"
#include "picongpu/particles/atomicPhysics/stage/BinElectrons.hpp"
#include "picongpu/particles/atomicPhysics/stage/CalculateStepLength.hpp"
#include "picongpu/particles/atomicPhysics/stage/CheckForOverSubscription.hpp"
#include "picongpu/particles/atomicPhysics/stage/CheckPresence.hpp"
#include "picongpu/particles/atomicPhysics/stage/ChooseTransition.hpp"
#include "picongpu/particles/atomicPhysics/stage/ChooseTransitionGroup.hpp"
#include "picongpu/particles/atomicPhysics/stage/DecelerateElectrons.hpp"
#include "picongpu/particles/atomicPhysics/stage/FillRateCache.hpp"
#include "picongpu/particles/atomicPhysics/stage/FixAtomicState.hpp"
#include "picongpu/particles/atomicPhysics/stage/LoadAtomicInputData.hpp"
#include "picongpu/particles/atomicPhysics/stage/RecordChanges.hpp"
#include "picongpu/particles/atomicPhysics/stage/RecordSuggestedChanges.hpp"
#include "picongpu/particles/atomicPhysics/stage/ResetAcceptedStatus.hpp"
#include "picongpu/particles/atomicPhysics/stage/ResetRateCache.hpp"
#include "picongpu/particles/atomicPhysics/stage/ResetSharedResources.hpp"
#include "picongpu/particles/atomicPhysics/stage/ResetTimeStepField.hpp"
#include "picongpu/particles/atomicPhysics/stage/RollForOverSubscription.hpp"
#include "picongpu/particles/atomicPhysics/stage/SpawnIonizationElectrons.hpp"
#include "picongpu/particles/atomicPhysics/stage/UpdateElectricField.hpp"
#include "picongpu/particles/atomicPhysics/stage/UpdateTimeRemaining.hpp"
#include "picongpu/particles/filter/filter.hpp"

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
        namespace enums
        {
            enum struct Loop : uint8_t
            {
                SubStep,
                ChooseTransition,
                RejectOverSubscription
            };
        } // namespace enums

        namespace debug = picongpu::atomicPhysics::debug;
        namespace localHelperFields = picongpu::particles::atomicPhysics::localHelperFields;
        namespace electronDistribution = picongpu::particles::atomicPhysics::electronDistribution;

        /** atomic physics stage
         *
         * models excited atomic state and ionization dynamics
         *gi
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
            using LinearizedBox = DataBoxDim1Access<typename T_Field::DataBoxType>;

            using OverSubscribedField = localHelperFields::SharedResourcesOverSubscribedField<picongpu::MappingDesc>;
            using TimeRemainingField = localHelperFields::TimeRemainingField<picongpu::MappingDesc>;
            using FoundUnboundField = localHelperFields::FoundUnboundIonField<picongpu::MappingDesc>;

            using BinSelection = electronDistribution::enums::BinSelection;

            //! species lists
            //!@{
            using AtomicPhysicsElectronSpecies = T_AtomicPhysicsElectronSpecies;
            using OnlyIPDElectronSpecies = T_OnlyIPDElectronSpecies;
            using AtomicPhysicsIonSpecies = T_AtomicPhysicsIonSpecies;
            using OnlyIPDIonSpecies = T_OnlyIPDIonSpecies;

            //! list of all electron species for IPD
            using IPDElectronSpecies = MakeSeq_t<AtomicPhysicsElectronSpecies, OnlyIPDElectronSpecies>;
            //! list of all ion species for IPD
            using IPDIonSpecies = MakeSeq_t<AtomicPhysicsIonSpecies, OnlyIPDIonSpecies>;
            //!@}

            //! debug print to console
            //!@{

            //! control multiple location debug prints depending on debug output setting
            template<enums::Loop T_Loop>
            static constexpr bool debugPrintActive()
            {
                constexpr bool isSubStepLoop = (T_Loop == enums::Loop::SubStep);
                constexpr bool isChooseTransition = (T_Loop == enums::Loop::ChooseTransition);
                constexpr bool isRejectOverSubscription = (T_Loop == enums::Loop::RejectOverSubscription);

                constexpr bool isActive = (isSubStepLoop)
                    || (isChooseTransition && debug::kernel::recordSuggestedChanges::PRINT_DEBUG_TO_CONSOLE)
                    || (isRejectOverSubscription && debug::kernel::rollForOverSubscription::PRINT_DEBUG_TO_CONSOLE);

                return isActive;
            }

            //! print electron histogram to console, debug only
            template<BinSelection T_BinSelection, enums::Loop T_Loop>
            HINLINE static void printHistogramToConsole(picongpu::MappingDesc const& mappingDesc, std::string name)
            {
                constexpr bool printActive = debug::electronHistogram::PRINT_TO_CONSOLE && debugPrintActive<T_Loop>();
                if constexpr(printActive)
                {
                    std::cout << name << std::endl;
                    picongpu::particles::atomicPhysics::stage::DumpSuperCellDataToConsole<
                        electronDistribution::
                            LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>,
                        electronDistribution::PrintHistogramToConsole<T_BinSelection>>{}(
                        mappingDesc,
                        "Electron_HistogramField");
                }
            }

            //! print SharedResourcesOverSubscribedField to console, debug only
            HINLINE static void printOverSubscriptionFieldToConsole(picongpu::MappingDesc const& mappingDesc)
            {
                if constexpr(debug::rejectionProbabilityCache::PRINT_TO_CONSOLE)
                    picongpu::particles::atomicPhysics::stage::DumpSuperCellDataToConsole<
                        localHelperFields::SharedResourcesOverSubscribedField<picongpu::MappingDesc>,
                        localHelperFields::PrintOverSubcriptionFieldToConsole>{}(
                        mappingDesc,
                        "SharedResourcesOverSubscribedField");
            }

            //! print rejectionProbabilityCache to console, debug only
            HINLINE static void printRejectionProbabilityCacheToConsole(picongpu::MappingDesc const& mappingDesc)
            {
                if constexpr(debug::rejectionProbabilityCache::PRINT_TO_CONSOLE)
                {
                    picongpu::particles::atomicPhysics::stage::DumpSuperCellDataToConsole<
                        localHelperFields::RejectionProbabilityCacheField<picongpu::MappingDesc>,
                        localHelperFields::PrintRejectionProbabilityCacheToConsole<true>>{}(
                        mappingDesc,
                        "RejectionProbabilityCacheField");
                }
            }

            //! print local time remaining to console, debug only
            HINLINE static void printTimeRemainingToConsole(picongpu::MappingDesc const& mappingDesc)
            {
                if constexpr(debug::timeRemaining::PRINT_TO_CONSOLE)
                {
                    picongpu::particles::atomicPhysics::stage::DumpSuperCellDataToConsole<
                        localHelperFields::TimeRemainingField<picongpu::MappingDesc>,
                        localHelperFields::PrintTimeRemaingToConsole>{}(mappingDesc, "TimeRemainingField");
                }
            }

            //! print local time step to console, debug only
            HINLINE static void printTimeStepToConsole(picongpu::MappingDesc const& mappingDesc)
            {
                if constexpr(debug::timeStep::PRINT_TO_CONSOLE)
                    picongpu::particles::atomicPhysics::stage::DumpSuperCellDataToConsole<
                        localHelperFields::TimeStepField<picongpu::MappingDesc>,
                        localHelperFields::PrintTimeStepToConsole>{}(mappingDesc, "TimeStepField");
            }

            //! print local fieldEnergyUseCache to console, debug only
            template<enums::Loop T_Loop>
            HINLINE static void printFieldEnergyUseCacheToConsole(picongpu::MappingDesc const& mappingDesc)
            {
                constexpr bool isActive = debug::fieldEnergyUseCache::PRINT_TO_CONSOLE && debugPrintActive<T_Loop>();
                if constexpr(isActive)
                    picongpu::particles::atomicPhysics::stage::DumpSuperCellDataToConsole<
                        localHelperFields::FieldEnergyUseCacheField<picongpu::MappingDesc>,
                        localHelperFields::PrintFieldEnergyUseCacheToConsole>{}(
                        mappingDesc,
                        "FieldEnergyUseCacheField");
            }
            //!@}

            //! set timeRemaining to PIC-time step
            HINLINE static void setTimeRemaining()
            {
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
                auto& localTimeRemainingField = *dc.get<TimeRemainingField>("TimeRemainingField");
                localTimeRemainingField.getDeviceBuffer().setValue(picongpu::sim.pic.getDt()); // sim.unit.time()
            }

            //! reset the histogram on device side
            HINLINE static void resetElectronEnergyHistogram()
            {
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
                auto& electronHistogramField = *dc.get<electronDistribution::LocalHistogramField<
                    picongpu::atomicPhysics::ElectronHistogram,
                    picongpu::MappingDesc>>("Electron_HistogramField");
                electronHistogramField.getDeviceBuffer().setValue(picongpu::atomicPhysics::ElectronHistogram());
            }

            //! reset foundUnboundIonField on device side
            HINLINE static void resetFoundUnboundIon()
            {
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
                auto& foundUnboundIonField = *dc.get<FoundUnboundField>("FoundUnboundIonField");
                foundUnboundIonField.getDeviceBuffer().setValue(0._X);
            };

            HINLINE static void resetAcceptStatus(picongpu::MappingDesc const& mappingDesc)
            {
                // particle[accepted_] = false, in each macro ion
                using ForEachIonSpeciesResetAcceptedStatus = pmacc::meta::ForEach<
                    AtomicPhysicsIonSpecies,
                    particles::atomicPhysics::stage::ResetAcceptedStatus<boost::mpl::_1>>;
                ForEachIonSpeciesResetAcceptedStatus{}(mappingDesc);
            }

            HINLINE static void debugForceConstantElectronTemperature([[maybe_unused]] uint32_t const currentStep)
            {
                if constexpr(debug::scFlyComparison::FORCE_CONSTANT_ELECTRON_TEMPERATURE)
                {
                    using ForEachElectronSpeciesSetTemperature = pmacc::meta::ForEach<
                        AtomicPhysicsElectronSpecies,
                        picongpu::particles::
                            Manipulate<picongpu::particles::atomicPhysics::SetTemperature, boost::mpl::_1>>;
                    ForEachElectronSpeciesSetTemperature{}(currentStep);
                };
            }

            HINLINE static void binElectronsToEnergyHistogram(picongpu::MappingDesc const& mappingDesc)
            {
                using ForEachElectronSpeciesBinElectrons = pmacc::meta::ForEach<
                    AtomicPhysicsElectronSpecies,
                    particles::atomicPhysics::stage::BinElectrons<boost::mpl::_1>>;
                ForEachElectronSpeciesBinElectrons{}(mappingDesc);

                printHistogramToConsole<BinSelection::All, enums::Loop::SubStep>(mappingDesc, "[after binning]");
            }

            //! calculate ionization potential depression parameters for every superCell
            HINLINE static void calculateIPDInput(picongpu::MappingDesc const& mappingDesc, uint32_t const currentStep)
            {
                picongpu::atomicPhysics::IPDModel::
                    template calculateIPDInput<T_numberAtomicPhysicsIonSpecies, IPDIonSpecies, IPDElectronSpecies>(
                        mappingDesc,
                        currentStep);
            }

            //! reset each superCell's time step
            HINLINE static void resetTimeStep(picongpu::MappingDesc const& mappingDesc)
            {
                // timeStep = timeRemaining
                picongpu::particles::atomicPhysics::stage::ResetTimeStepField<T_numberAtomicPhysicsIonSpecies>()(
                    mappingDesc);
            }

            //! reset each superCell's rate cache
            HINLINE static void resetRateCache()
            {
                using ForEachIonSpeciesResetRateCache = pmacc::meta::
                    ForEach<AtomicPhysicsIonSpecies, particles::atomicPhysics::stage::ResetRateCache<boost::mpl::_1>>;
                ForEachIonSpeciesResetRateCache{}();
            }

            //! check which atomic states are actually present in each superCell
            HINLINE static void checkPresence(picongpu::MappingDesc const& mappingDesc)
            {
                using ForEachIonSpeciesCheckPresenceOfAtomicStates = pmacc::meta::
                    ForEach<AtomicPhysicsIonSpecies, particles::atomicPhysics::stage::CheckPresence<boost::mpl::_1>>;
                ForEachIonSpeciesCheckPresenceOfAtomicStates{}(mappingDesc);
            }

            //! fill each superCell's rate cache
            HINLINE static void fillRateCache(picongpu::MappingDesc const& mappingDesc)
            {
                using ForEachIonSpeciesFillRateCache = pmacc::meta::
                    ForEach<AtomicPhysicsIonSpecies, particles::atomicPhysics::stage::FillRateCache<boost::mpl::_1>>;
                ForEachIonSpeciesFillRateCache{}(mappingDesc);

                using ForEachIonSpeciesDumpRateCacheToConsole = pmacc::meta::ForEach<
                    AtomicPhysicsIonSpecies,
                    particles::atomicPhysics::stage::DumpRateCacheToConsole<boost::mpl::_1>>;

                if constexpr(debug::rateCache::PRINT_TO_CONSOLE)
                    ForEachIonSpeciesDumpRateCacheToConsole{}(mappingDesc);
            }

            //! min(1/(-R_ii)) * alpha, calculate local atomicPhysics time step length
            HINLINE static void calculateSubStepLength(picongpu::MappingDesc const& mappingDesc)
            {
                using ForEachIonSpeciesCalculateStepLength = pmacc::meta::ForEach<
                    AtomicPhysicsIonSpecies,
                    particles::atomicPhysics::stage::CalculateStepLength<boost::mpl::_1>>;
                ForEachIonSpeciesCalculateStepLength{}(mappingDesc);
            }

            HINLINE static void chooseTransition(picongpu::MappingDesc const& mappingDesc, uint32_t const currentStep)
            {
                // randomly roll transition for each not yet accepted macro ion
                using ForEachIonSpeciesChooseTransitionGroup = pmacc::meta::ForEach<
                    AtomicPhysicsIonSpecies,
                    particles::atomicPhysics::stage::ChooseTransitionGroup<boost::mpl::_1>>;
                ForEachIonSpeciesChooseTransitionGroup{}(mappingDesc, currentStep);

                using ForEachIonSpeciesChooseTransition = pmacc::meta::ForEach<
                    AtomicPhysicsIonSpecies,
                    particles::atomicPhysics::stage::ChooseTransition<boost::mpl::_1>>;
                ForEachIonSpeciesChooseTransition{}(mappingDesc, currentStep);
            }

            //! record all shared resources usage by accepted transitions
            HINLINE static void recordSuggestedChanges(picongpu::MappingDesc const& mappingDesc)
            {
                picongpu::particles::atomicPhysics::stage::ResetSharedResources<T_numberAtomicPhysicsIonSpecies>{}(
                    mappingDesc);
                using ForEachIonSpeciesRecordSuggestedChanges = pmacc::meta::ForEach<
                    AtomicPhysicsIonSpecies,
                    particles::atomicPhysics::stage::RecordSuggestedChanges<boost::mpl::_1>>;
                ForEachIonSpeciesRecordSuggestedChanges{}(mappingDesc);
            }

            // check if an electron histogram bin () is over subscription --> superCellOversubScriptionField
            template<
                enums::Loop T_Loop,
                typename T_SuperCellSharedResourcesOverSubscriptionField,
                typename T_DeviceReduce>
            HINLINE static bool isASharedResourceOverSubscribed(
                picongpu::MappingDesc const& mappingDesc,
                T_SuperCellSharedResourcesOverSubscriptionField& perSuperCellSharedResourcesOverSubscriptionField,
                T_DeviceReduce& deviceReduce)
            {
                DataSpace<picongpu::simDim> const fieldGridLayoutOverSubscription
                    = perSuperCellSharedResourcesOverSubscriptionField.getGridLayout().sizeWithoutGuardND();

                picongpu::particles::atomicPhysics::stage::CheckForOverSubscription<T_numberAtomicPhysicsIonSpecies>{}(
                    mappingDesc);

                auto linearizedOverSubscribedBox = LinearizedBox<OverSubscribedField>(
                    perSuperCellSharedResourcesOverSubscriptionField.getDeviceDataBox(),
                    fieldGridLayoutOverSubscription);

                bool isOverSubscribed = static_cast<bool>(deviceReduce(
                    pmacc::math::operation::Or(),
                    linearizedOverSubscribedBox,
                    fieldGridLayoutOverSubscription.productOfComponents()));

                // debug only
                std::string message;
                if constexpr(debugPrintActive<T_Loop>())
                {
                    message += "[histogram oversubscribed?]: ";
                    message += (isOverSubscribed ? "true" : "false");
                }
                printHistogramToConsole<BinSelection::OnlyOverSubscribed, T_Loop>(mappingDesc, message);
                printFieldEnergyUseCacheToConsole<T_Loop>(mappingDesc);
                printOverSubscriptionFieldToConsole(mappingDesc);
                printRejectionProbabilityCacheToConsole(mappingDesc);

                // check whether a least one histogram is oversubscribed
                return isOverSubscribed;
            }

            HINLINE static void randomlyRejectTransitionFromOverSubscribedResources(
                picongpu::MappingDesc const& mappingDesc,
                uint32_t const currentStep)
            {
                using ForEachIonSpeciesRollForOverSubscription = pmacc::meta::ForEach<
                    AtomicPhysicsIonSpecies,
                    particles::atomicPhysics::stage::RollForOverSubscription<boost::mpl::_1>>;
                ForEachIonSpeciesRollForOverSubscription{}(mappingDesc, currentStep);
            }

            /** update atomic state and accumulate delta energy for delta energy histogram
             *
             * @note may already update the atomic state since the following kernels DecelerateElectrons and
             * SpawnIonizationElectrons only use the transitionIndex particle attribute */
            HINLINE static void recordChanges(picongpu::MappingDesc const& mappingDesc)
            {
                using ForEachIonSpeciesRecordChanges = pmacc::meta::
                    ForEach<AtomicPhysicsIonSpecies, particles::atomicPhysics::stage::RecordChanges<boost::mpl::_1>>;
                ForEachIonSpeciesRecordChanges{}(mappingDesc);
            }

            HINLINE static void updateElectrons(picongpu::MappingDesc const& mappingDesc, uint32_t const currentStep)
            {
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
            }

            HINLINE static void updateElectricField(picongpu::MappingDesc const& mappingDesc)
            {
                picongpu::particles::atomicPhysics::stage::UpdateElectricField<T_numberAtomicPhysicsIonSpecies>()(
                    mappingDesc);
            }

            template<typename T_DeviceReduce>
            HINLINE static void doIPDIonization(
                picongpu::MappingDesc const& mappingDesc,
                uint32_t const currentStep,
                T_DeviceReduce& deviceReduce)
            {
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

                auto& foundUnboundIonField = *dc.get<FoundUnboundField>("FoundUnboundIonField");
                DataSpace<picongpu::simDim> const fieldGridLayoutFoundUnbound
                    = foundUnboundIonField.getGridLayout().sizeWithoutGuardND();

                // pressure ionization loop, ends when no ion in unbound state anymore
                bool foundUnbound = true;
                do
                {
                    resetFoundUnboundIon();
                    calculateIPDInput(mappingDesc, currentStep);
                    picongpu::atomicPhysics::IPDModel::template applyIPDIonization<AtomicPhysicsIonSpecies>(
                        mappingDesc,
                        currentStep);

                    auto linearizedFoundUnboundIonBox = LinearizedBox<FoundUnboundField>(
                        foundUnboundIonField.getDeviceDataBox(),
                        fieldGridLayoutFoundUnbound);

                    foundUnbound = static_cast<bool>(deviceReduce(
                        pmacc::math::operation::Or(),
                        linearizedFoundUnboundIonBox,
                        fieldGridLayoutFoundUnbound.productOfComponents()));
                } // end pressure ionization loop
                while(foundUnbound);
            }

            HINLINE static void updateTimeRemaining(picongpu::MappingDesc const& mappingDesc)
            {
                // timeRemaining -= timeStep
                picongpu::particles::atomicPhysics::stage::UpdateTimeRemaining<T_numberAtomicPhysicsIonSpecies>()(
                    mappingDesc);
            }

            template<typename T_DeviceReduce>
            HINLINE static bool isSubSteppingFinished(
                picongpu::MappingDesc const& mappingDesc,
                T_DeviceReduce& deviceReduce)
            {
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
                auto& localTimeRemainingField = *dc.get<TimeRemainingField>("TimeRemainingField");
                DataSpace<picongpu::simDim> const fieldGridLayoutTimeRemaining
                    = localTimeRemainingField.getGridLayout().sizeWithoutGuardND();

                auto linearizedTimeRemainingBox = LinearizedBox<TimeRemainingField>(
                    localTimeRemainingField.getDeviceDataBox(),
                    fieldGridLayoutTimeRemaining);

                return deviceReduce(
                           pmacc::math::operation::Max(),
                           linearizedTimeRemainingBox,
                           fieldGridLayoutTimeRemaining.productOfComponents())
                    <= 0._X;
            }

        public:
            AtomicPhysics() = default;

            //! atomic physics stage sub-stage calls
            void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
            {
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

                auto& perSuperCellSharedResourcesOverSubscribedField
                    = *dc.get<OverSubscribedField>("SharedResourcesOverSubscribedField");

                /// @todo find better way than hard code old value, Brian Marre, 2023
                // `static` avoids that reduce is allocating each time step memory, which will reduce the performance.
                static pmacc::device::Reduce deviceLocalReduce = pmacc::device::Reduce(static_cast<uint32_t>(1200u));

                setTimeRemaining(); // = (Delta t)_PIC
                // fix atomic state and charge state inconsistency
                using ForEachIonSpeciesFixAtomicState = pmacc::meta::
                    ForEach<AtomicPhysicsIonSpecies, particles::atomicPhysics::stage::FixAtomicState<boost::mpl::_1>>;
                ForEachIonSpeciesFixAtomicState{}(mappingDesc);

                // atomicPhysics sub-stepping loop
                bool isSubSteppingComplete = false;
                while(!isSubSteppingComplete)
                {
                    resetAcceptStatus(mappingDesc);
                    resetElectronEnergyHistogram();
                    debugForceConstantElectronTemperature(currentStep);
                    doIPDIonization(mappingDesc, currentStep, deviceLocalReduce);
                    binElectronsToEnergyHistogram(mappingDesc);
                    resetTimeStep(mappingDesc);
                    resetRateCache();
                    checkPresence(mappingDesc);
                    fillRateCache(mappingDesc);
                    calculateSubStepLength(mappingDesc);

                    // choose transition loop
                    bool isHistogramOverSubscribed = true;
                    while(isHistogramOverSubscribed)
                    {
                        chooseTransition(mappingDesc, currentStep);
                        recordSuggestedChanges(mappingDesc);

                        bool isOverSubscribed = isASharedResourceOverSubscribed<enums::Loop::ChooseTransition>(
                            mappingDesc,
                            perSuperCellSharedResourcesOverSubscribedField,
                            deviceLocalReduce);
                        isHistogramOverSubscribed = isOverSubscribed;

                        while(isOverSubscribed)
                        {
                            // at least one superCell electron histogram over-subscribed
                            randomlyRejectTransitionFromOverSubscribedResources(mappingDesc, currentStep);
                            recordSuggestedChanges(mappingDesc);

                            isOverSubscribed = isASharedResourceOverSubscribed<enums::Loop::RejectOverSubscription>(
                                mappingDesc,
                                perSuperCellSharedResourcesOverSubscribedField,
                                deviceLocalReduce);
                        } // end remove over subscription loop

                        if constexpr(debug::kernel::rollForOverSubscription::PRINT_DEBUG_TO_CONSOLE)
                            std::cout << "[rejection loop complete]" << std::endl;
                    } // end choose transition loop

                    printTimeRemainingToConsole(mappingDesc);
                    printTimeStepToConsole(mappingDesc);
                    printFieldEnergyUseCacheToConsole<enums::Loop::SubStep>(mappingDesc);

                    recordChanges(mappingDesc);
                    updateElectrons(mappingDesc, currentStep);
                    updateElectricField(mappingDesc);
                    updateTimeRemaining(mappingDesc);
                    isSubSteppingComplete = isSubSteppingFinished(mappingDesc, deviceLocalReduce);
                } // end atomicPhysics sub-stepping loop

                // ensure no unbound states are visible to the rest of the loop
                doIPDIonization(mappingDesc, currentStep, deviceLocalReduce);
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


    void AtomicPhysics::loadAtomicInputData(DataConnector& dataConnector)
    {
        pmacc::meta::ForEach<
            SpeciesRepresentingAtomicPhysicsIons,
            particles::atomicPhysics::stage::LoadAtomicInputData<boost::mpl::_1>>
            ForEachIonSpeciesLoadAtomicInputData;
        ForEachIonSpeciesLoadAtomicInputData(dataConnector);
    }


    AtomicPhysics::AtomicPhysics(picongpu::MappingDesc const mappingDesc)
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

    void AtomicPhysics::fixAtomicStateInit(picongpu::MappingDesc const mappingDesc)
    {
        using ForEachIonSpeciesFixAtomicState = pmacc::meta::ForEach<
            SpeciesRepresentingAtomicPhysicsIons,
            particles::atomicPhysics::stage::FixAtomicState<boost::mpl::_1>>;
        ForEachIonSpeciesFixAtomicState{}(mappingDesc);
    }

    void AtomicPhysics::operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
    {
        if constexpr(atomicPhysicsActive)
        {
            //! list of all species of macro particles that partake in atomicPhysics as ions
            using AtomicPhysicsIonSpecies = SpeciesRepresentingAtomicPhysicsIons;
            //! list of all only IPD partaking ion species
            using OnlyIPDIonSpecies = particles::atomicPhysics::traits::
                FilterByParticleType_t<VectorAllSpecies, picongpu::particles::atomicPhysics::Tags::OnlyIPDIon>;

            //! list of all species of macro particles that partake in atomicPhysics as electrons
            using AtomicPhysicsElectronSpecies = particles::atomicPhysics::traits::
                FilterByParticleType_t<VectorAllSpecies, picongpu::particles::atomicPhysics::Tags::Electron>;

            //! list of all only IPD partaking electron species
            using OnlyIPDElectronSpecies = particles::atomicPhysics::traits::
                FilterByParticleType_t<VectorAllSpecies, picongpu::particles::atomicPhysics::Tags::OnlyIPDElectron>;

            detail::AtomicPhysics<
                AtomicPhysicsIonSpecies,
                OnlyIPDIonSpecies,
                AtomicPhysicsElectronSpecies,
                OnlyIPDElectronSpecies,
                numberAtomicPhysicsIonSpecies>{}(mappingDesc, currentStep);
        }
    }
} // namespace picongpu::simulation::stage
