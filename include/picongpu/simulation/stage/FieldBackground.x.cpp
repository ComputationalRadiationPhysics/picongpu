/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "picongpu/simulation/stage/FieldBackground.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/background/cellwiseOperation.hpp"
#include "picongpu/param/fieldBackground.param"

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
                struct ApplyFieldBackground : public detail::IApplyFieldBackground
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
                    void add(uint32_t const step) override
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
                    void subtract(uint32_t const step) override
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

            void FieldBackground::init(MappingDesc const cellDescription)
            {
                //! Implememtation type to apply background to field E
                using ApplyE = detail::ApplyFieldBackground<FieldE, FieldBackgroundE>;
                this->applyE = std::make_unique<ApplyE>(cellDescription, duplicateFields);

                //! Implememtation type to apply background to field B
                using ApplyB = detail::ApplyFieldBackground<FieldB, FieldBackgroundB>;
                this->applyB = std::make_unique<ApplyB>(cellDescription, duplicateFields);
            }

            /** Add field background to the electromagnetic field
             *
             * Affects data sets named FieldE::getName(), FieldB::getName().
             * As the result of this operation, they will have a sum of old values and background values.
             *
             * @param step index of time iteration
             */
            void FieldBackground::add(uint32_t const step)
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
            void FieldBackground::subtract(uint32_t const step)
            {
                checkInitialization();
                applyE->subtract(step);
                applyB->subtract(step);
            }

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
