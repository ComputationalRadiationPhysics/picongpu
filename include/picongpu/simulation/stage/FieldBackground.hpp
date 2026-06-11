/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera, Richard Pausch, Alexander Debus, Marco Garten, Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/background/cellwiseOperation.hpp"
#include "picongpu/fields/background/param.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/type/Area.hpp>

#include <boost/program_options.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            namespace detail
            {
                /* Implementation of background application to the given field
                 *
                 * @tparam T_Field field affected, e.g. picongpu::FieldE
                 * @tparam T_FieldBackground field background to apply, e.g. picongpu::FieldBackgroundE
                 */
                template<typename T_Field, typename T_FieldBackground>
                class ApplyFieldBackground
                {
                public:
                    //! Field affected
                    using Field = T_Field;

                    //! Field background to apply
                    using FieldBackground = T_FieldBackground;

                    /** Create an object to apply the background
                     *
                     * @param cellDescription mapping for kernels
                     * @param useDuplicateField flag to store duplicate of the field
                     */
                    ApplyFieldBackground(MappingDesc const cellDescription, bool const useDuplicateField)
                        : isEnabled(FieldBackground::InfluenceParticlePusher)
                        , useDuplicateField(useDuplicateField)
                        , restoreFromDuplicateField(false)
                        , cellDescription(cellDescription)
                    {
                        if(isEnabled && useDuplicateField)
                        {
                            // Allocate a duplicate field buffer and copy the values
                            DataConnector& dc = Environment<>::get().DataConnector();
                            auto field = dc.get<Field>(Field::getName());
                            auto const& gridBuffer = field->getGridBuffer();
                            duplicateBuffer = pmacc::makeDeepCopy(gridBuffer.getDeviceBuffer());
                        }
                    }

                    /** Add the field background in the whole local domain
                     *
                     * @param step index of time iteration
                     */
                    void add(uint32_t const step)
                    {
                        if(!isEnabled)
                            return;
                        DataConnector& dc = Environment<>::get().DataConnector();
                        auto& field = *dc.get<Field>(Field::getName());
                        // Always add to the field, conditionally make a copy of the old values first
                        if(useDuplicateField)
                        {
                            auto& gridBuffer = field.getGridBuffer();
                            duplicateBuffer->copyFrom(gridBuffer.getDeviceBuffer());
                            restoreFromDuplicateField = true;
                        }
                        apply(step, pmacc::math::operation::Add(), field);
                    }

                    /** Subtract the field background in the whole local domain
                     *
                     * @param step index of time iteration
                     */
                    void subtract(uint32_t const step)
                    {
                        if(!isEnabled)
                            return;
                        DataConnector& dc = Environment<>::get().DataConnector();
                        auto& field = *dc.get<Field>(Field::getName());
                        /* Either restore from the pre-made copy or subtract.
                         * Note that here it is not sufficient to check for useDuplicateField as it
                         * is not necessarily up-to-date, e.g. right after loading from a checkpoint.
                         */
                        if(restoreFromDuplicateField)
                        {
                            auto& gridBuffer = field.getGridBuffer();
                            gridBuffer.getDeviceBuffer().copyFrom(*duplicateBuffer);
                            restoreFromDuplicateField = false;
                        }
                        else
                            apply(step, pmacc::math::operation::Sub(), field);
                    }

                private:
                    //! Is the field background enabled
                    bool isEnabled;

                    //! Flag to store duplicate of field when the background is enabled
                    bool useDuplicateField;

                    //! Flag to restore from the duplicate field: true if it is enabled and up-to-date
                    bool restoreFromDuplicateField;

                    //! Buffer type to store duplicated values
                    using DeviceBuffer = typename Field::Buffer::DBuffer;

                    //! Buffer to store duplicated values, only used when useDuplicateField is true
                    std::unique_ptr<DeviceBuffer> duplicateBuffer;

                    //! Mapping for kernels
                    MappingDesc const cellDescription;

                    /** Apply the given functor to the field background in the whole local domain
                     *
                     * @tparam T_Functor functor type compatible to pmacc::math::operation
                     *
                     * @param step index of time iteration
                     * @param functor functor to apply
                     * @param field field object which data is modified
                     */
                    template<typename T_Functor>
                    void apply(uint32_t const step, T_Functor functor, Field& field)
                    {
                        constexpr auto area = CORE + BORDER + GUARD;
                        using CallBackground = cellwiseOperation::CellwiseOperation<area>;
                        CallBackground callBackground(cellDescription);
                        callBackground(&field, functor, FieldBackground(field.getUnit()), step);
                    }
                };
            } // namespace detail

            //! Functor for the stage of the PIC loop applying field background
            class FieldBackground : public ISimulationData
            {
            public:
                /** Register program options for field background
                 *
                 * @param desc program options following boost::program_options::options_description
                 */
                void registerHelp(po::options_description& desc)
                {
                    desc.add_options()(
                        "fieldBackground.duplicateFields",
                        po::value<bool>(&duplicateFields)->zero_tokens(),
                        "duplicate E and B field storage inside field background to improve its performance "
                        "and potentially avoid some numerical noise at cost of using more memory, "
                        "only affects the fields with activated background")(
                        "fieldBackground.influencesPlugins",
                        po::value<bool>(&influencesPlugins)->default_value(true),
                        "enable the field background to be seen by the plugins")(
                        "fieldBackground.influencesDumps",
                        po::value<bool>(&influencesDumps)->default_value(true),
                        "enable the field background to be included in dumps (incl. checkpoints).");
                }

                /** Initialize field background stage
                 *
                 * This method must be called once before calling add(), subtract() and fillSimulation().
                 * The initialization has to be delayed for this class as it needs registerHelp() like the plugins do.
                 *
                 * @param cellDescription mapping for kernels
                 */
                void init(MappingDesc const cellDescription)
                {
                    applyE = std::make_unique<ApplyE>(cellDescription, duplicateFields);
                    applyB = std::make_unique<ApplyB>(cellDescription, duplicateFields);
                }

                /** Enable field background if not already enabled
                 *
                 * Adds the field background if it hasn't been applied yet. This ensures the background is only added
                 * once regardless of multiple calls to this method.
                 *
                 * @param step index of time iteration
                 */
                void enable(uint32_t const step)
                {
                    if(!appliedState)
                    {
                        add(step);
                        appliedState = true;
                    }
                }

                /** Disable field background if currently enabled
                 *
                 * Removes the field background if it is currently applied. This ensures the background is only
                 * subtracted once regardless of multiple calls to this method.
                 *
                 * @param step index of time iteration
                 */
                void disable(uint32_t const step)
                {
                    if(appliedState)
                    {
                        subtract(step);
                        appliedState = false;
                    }
                }

                /** Set field background state for plugin processing
                 *
                 * Plugins see the background only if influencesPlugins is true.
                 *
                 * @param step index of time iteration
                 */
                void toPluginState(uint32_t const step)
                {
                    influencesPlugins ? enable(step) : disable(step);
                }

                /** Set field background state for dumps/checkpoints
                 *
                 * Dumps include the background only if influencesDumps is true
                 *
                 * @param step index of time iteration
                 */
                void toDumpState(uint32_t const step)
                {
                    influencesDumps ? enable(step) : disable(step);
                }

                /** Set field background state on restart
                 *
                 * Sets the internal state to indicate that the field background has already
                 * been applied. This should be called when restarting. If restarting from a checkpoint where
                 * the fields already contain the background values, this prevents the background
                 * from being added twice.
                 *
                 * @param oldDumpState Whether the field background was already included in the checkpoint
                 */
                void restart(bool const oldDumpState)
                {
                    if(oldDumpState)
                    {
                        appliedState = true;
                    }
                }

                bool getInfluencesDumps() const
                {
                    return influencesDumps;
                }

                /**
                 * Return the globally unique identifier for this simulation data.
                 *
                 * @return globally unique identifier
                 */
                SimulationDataId getUniqueId() override
                {
                    return "FieldBackground";
                }

                /**
                 * Synchronizes simulation data, meaning accessing (host side) data
                 * will return up-to-date values.
                 */
                void synchronize() override {};

            private:
                /** Add field background to the electromagnetic field
                 *
                 * Affects data sets named FieldE::getName(), FieldB::getName().
                 * As the result of this operation, they will have a sum of old values and background values.
                 *
                 * @param step index of time iteration
                 */
                void add(uint32_t const step)
                {
                    checkInitialization();
                    applyE->add(step);
                    applyB->add(step);
                }

                /** Subtract field background from the electromagnetic field
                 *
                 * Affects data sets named FieldE::getName(), FieldB::getName().
                 * As the result of this operation, they will have values like before the last call to add().
                 *
                 * Warning: when fieldBackground.duplicateFields is enabled, the fields are assumed to not have changed
                 * since the call to add(). Having fieldBackground.duplicateFields disabled does not rely on this.
                 * However, this assumption should generally hold true in the PIC computational loop.
                 *
                 * @param step index of time iteration
                 */
                void subtract(uint32_t const step)
                {
                    checkInitialization();
                    applyE->subtract(step);
                    applyB->subtract(step);
                }

                //! Check if this class was properly initialized, throws when failed
                void checkInitialization() const
                {
                    if(!applyE || !applyB)
                        throw std::runtime_error("simulation::stage::FieldBackground used without init() called");
                }

                //! Implememtation type to apply background to field E
                using ApplyE = detail::ApplyFieldBackground<FieldE, FieldBackgroundE>;

                //! Object to apply background to field E
                std::unique_ptr<ApplyE> applyE;

                //! Implememtation type to apply background to field B
                using ApplyB = detail::ApplyFieldBackground<FieldB, FieldBackgroundB>;

                //! Object to apply background to field B
                std::unique_ptr<ApplyB> applyB;

                //! Flag to store duplicates fields with enabled backgrounds
                bool duplicateFields = false;

                //! Flag to indicate if field background is seen by plugins
                bool influencesPlugins{true};
                //! Flag to indicate if field background is seen by dumps
                bool influencesDumps{true};
                //! Flag for state to indicate if field background is currently applied
                bool appliedState{false};
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
