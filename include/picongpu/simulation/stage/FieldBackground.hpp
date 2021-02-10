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

#include <cstdint>
#include <stdexcept>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            namespace detail
            {
                /* Functor to apply the given field background to the given field
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

                    /** Create a functor to apply the background
                     *
                     * @param cellDescription mapping for kernels
                     */
                    ApplyFieldBackground(MappingDesc const cellDescription)
                        : cellDescription(cellDescription)
                        , isEnabled(FieldBackground::InfluenceParticlePusher)
                    {
                    }

                    /** Apply the given functor to the field background in the given area
                     *
                     * @tparam T_area area to operate on
                     * @tparam T_Functor functor type compatible to pmacc::nvidia::functors
                     *
                     * @param step index of time iteration
                     * @param functor functor to apply
                     */
                    template<uint32_t T_area, typename T_Functor>
                    void operator()(uint32_t const step, T_Functor functor) const
                    {
                        if(!isEnabled)
                            return;
                        using namespace pmacc;
                        using CallBackground = cellwiseOperation::CellwiseOperation<T_area>;
                        CallBackground callBackground(cellDescription);
                        DataConnector& dc = Environment<>::get().DataConnector();
                        auto field = dc.get<Field>(Field::getName(), true);
                        callBackground(field, functor, FieldBackground(field->getUnit()), step);
                        dc.releaseData(Field::getName());
                    }

                private:
                    //! Is the field background enabled
                    bool isEnabled;

                    //! Mapping for kernels
                    MappingDesc const cellDescription;
                };
            } // namespace detail

            //! Functor for the stage of the PIC loop applying field background
            class FieldBackground
            {
            public:
                /** Register program options for field background
                 *
                 * @param desc boost::program_options::options_description
                 */
                void registerHelp(po::options_description& desc)
                {
                    desc.add_options()(
                        "fieldBackground.useExtraMemory",
                        po::value<bool>(&useExtraMemory)->zero_tokens(),
                        "use extra fields for field background to improve precision and potentially avoid some "
                        "numerical noise");
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
                    applyE = std::make_unique<ApplyE>(cellDescription);
                    applyB = std::make_unique<ApplyB>(cellDescription);
                }

                /** Add field background to the electromagnetic field
                 *
                 * Affects data sets named FieldE::getName(), FieldB::getName().
                 * As the result of this operation, they will have a sum of old values and background values.
                 *
                 * @param step index of time iteration
                 */
                void add(uint32_t const step) const
                {
                    apply<CORE + BORDER + GUARD>(step, pmacc::nvidia::functors::Add{});
                }

                /** Subtract field background from the electromagnetic field
                 *
                 * Affects data sets named FieldE::getName(), FieldB::getName().
                 * As the result of this operation, they will have values like before calling add().
                 *
                 * Warning: when fieldBackground.useExtraMemory is enabled, the fields are assumed to not have changed
                 * since the call to add(). Having fieldBackground.useExtraMemory disabled does not rely on this.
                 *
                 * @param step index of time iteration
                 */
                void subtract(uint32_t const step) const
                {
                    apply<CORE + BORDER + GUARD>(step, pmacc::nvidia::functors::Sub{});
                }

                /** Set field background to a consistent initial state for starting or resuming a simulation
                 *
                 * This method must be called during filling the simulation.
                 *
                 * @param step index of time iteration
                 */
                void fillSimulation(uint32_t const step) const
                {
                    /* restore background fields in GUARD
                     *
                     * loads the outer GUARDS of the global domain for absorbing/open boundary condtions
                     *
                     * @todo as soon as we add GUARD fields to the checkpoint data, e.g. for PML boundary
                     *       conditions, this section needs to be removed
                     */
                    apply<GUARD>(step, pmacc::nvidia::functors::Add());
                }

            private:
                /** Apply the given functor to E and B field backgrounds in the given area
                 *
                 * @tparam T_area area to operate on
                 * @tparam T_Functor functor type compatible to pmacc::nvidia::functors
                 *
                 * @param step index of time iteration
                 * @param functor functor to apply
                 */
                template<uint32_t T_area, typename T_Functor>
                void apply(uint32_t const step, T_Functor functor) const
                {
                    if(!applyE || !applyB)
                        throw std::runtime_error("simulation::stage::FieldBackground used without init() called");
                    applyE->operator()<T_area>(step, functor);
                    applyB->operator()<T_area>(step, functor);
                }

                //! Functor type to apply background to field E
                using ApplyE = detail::ApplyFieldBackground<FieldE, FieldBackgroundE>;

                //! Functor to apply background to field E
                std::unique_ptr<ApplyE> applyE;

                //! Functor type to apply background to field B
                using ApplyB = detail::ApplyFieldBackground<FieldB, FieldBackgroundB>;

                //! Functor to apply background to field B
                std::unique_ptr<ApplyB> applyB;

                //! Whether to use extra fields to store activated background fields
                bool useExtraMemory = false;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
