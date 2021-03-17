/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
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
#include "picongpu/fields/background/cellwiseOperation.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/nvidia/functors/Sub.hpp>
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
                        : cellDescription(cellDescription)
                        , useDuplicateField(useDuplicateField)
                        , restoreFromDuplicateField(false)
                        , isEnabled(FieldBackground::InfluenceParticlePusher)
                    {
                        if(isEnabled && useDuplicateField)
                        {
                            // Allocate a duplicate field buffer and copy the values
                            DataConnector& dc = Environment<>::get().DataConnector();
                            auto field = dc.get<Field>(Field::getName(), true);
                            auto const& gridBuffer = field->getGridBuffer();
                            duplicateBuffer = pmacc::makeDeepCopy(gridBuffer.getDeviceBuffer());
                            dc.releaseData(Field::getName());
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
                        auto& field = *dc.get<Field>(Field::getName(), true);
                        // Always add to the field, conditionally make a copy of the old values first
                        if(useDuplicateField)
                        {
                            auto& gridBuffer = field.getGridBuffer();
                            duplicateBuffer->copyFrom(gridBuffer.getDeviceBuffer());
                            restoreFromDuplicateField = true;
                        }
                        apply(step, pmacc::nvidia::functors::Add(), field);
                        dc.releaseData(Field::getName());
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
                        auto& field = *dc.get<Field>(Field::getName(), true);
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
                            apply(step, pmacc::nvidia::functors::Sub(), field);
                        dc.releaseData(Field::getName());
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
                     * @tparam T_Functor functor type compatible to pmacc::nvidia::functors
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
            class FieldBackground
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
                        "only affects the fields with activated background");
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

            private:
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
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
