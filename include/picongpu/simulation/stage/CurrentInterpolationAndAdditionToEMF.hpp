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

#include "picongpu/fields/currentInterpolation/CurrentInterpolation.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/type/Area.hpp>

#include <boost/mpl/count.hpp>
#include <boost/program_options/options_description.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the stage of the PIC loop performing current interpolation
             *  and addition to grid values of the electromagnetic field
             */
            class CurrentInterpolationAndAdditionToEMF
            {
            public:
                /** Register program options for current interpolation
                 *
                 * @param desc program options following boost::program_options::options_description
                 */
                void registerHelp(po::options_description& desc)
                {
                    desc.add_options()(
                        "currentInterpolation",
                        po::value<std::string>(&kindName),
                        (std::string("Current interpolation kind [None, Binomial] default: " + kindName).c_str()));
                }

                /** Initialize the current interpolation stage
                 *
                 * This method has to be called during initialization of the simulation.
                 * Before this method is called, the instance of CurrentInterpolation cannot be used safely.
                 */
                void init()
                {
                    using namespace fields::currentInterpolation;
                    auto& interpolation = CurrentInterpolation::get();
                    // So far there are only two kinds and so names are hardcoded
                    if(kindName == "none")
                        interpolation.kind = CurrentInterpolation::Kind::None;
                    else if(kindName == "binomial")
                        interpolation.kind = CurrentInterpolation::Kind::Binomial;
                    else
                        throw std::runtime_error("Unsupported current interpolation type");
                }

                /** Compute the current created by particles and add it to the current density
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const
                {
                    using namespace pmacc;
                    using SpeciesWithCurrentSolver =
                        typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, current<>>::type;
                    auto const numSpeciesWithCurrentSolver = bmpl::size<SpeciesWithCurrentSolver>::type::value;
                    auto const existsCurrent = numSpeciesWithCurrentSolver > 0;
                    if(!existsCurrent)
                        return;

                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto& fieldJ = *dc.get<FieldJ>(FieldJ::getName(), true);
                    auto eRecvCurrent = fieldJ.asyncCommunication(__getTransactionEvent());
                    auto& interpolation = fields::currentInterpolation::CurrentInterpolation::get();
                    auto const currentRecvLower = interpolation.getLowerMargin();
                    auto const currentRecvUpper = interpolation.getUpperMargin();

                    /* without interpolation, we do not need to access the FieldJ GUARD
                     * and can therefore overlap communication of GUARD->(ADD)BORDER & computation of CORE
                     */
                    if(currentRecvLower == DataSpace<simDim>::create(0)
                       && currentRecvUpper == DataSpace<simDim>::create(0))
                    {
                        addCurrentToEMF<type::CORE>(fieldJ);
                        __setTransactionEvent(eRecvCurrent);
                        addCurrentToEMF<type::BORDER>(fieldJ);
                    }
                    else
                    {
                        /* in case we perform a current interpolation/filter, we need
                         * to access the BORDER area from the CORE (and the GUARD area
                         * from the BORDER)
                         * `fieldJ->asyncCommunication` first adds the neighbors' values
                         * to BORDER (send) and then updates the GUARD (receive)
                         * \todo split the last `receive` part in a separate method to
                         *       allow already a computation of CORE */
                        __setTransactionEvent(eRecvCurrent);
                        addCurrentToEMF<type::CORE + type::BORDER>(fieldJ);
                    }
                    dc.releaseData(FieldJ::getName());
                }

            private:
                //! Name set by program option
                std::string kindName = "none";

                /* Call addCurrentToEMF method of fieldJ for the given area
                 *
                 * This function performs a transition from the run-time realm of CurrentInterpolation into the
                 * template realm of fieldJ.addCurrentToEMF() operating with interpolation functors.
                 *
                 * @tparam T_area area to operate once
                 *
                 * @param fieldJ object representing the current field
                 */
                template<std::uint32_t T_area>
                void addCurrentToEMF(FieldJ& fieldJ) const
                {
                    using namespace fields::currentInterpolation;
                    auto const kind = CurrentInterpolation::get().kind;
                    if(kind == CurrentInterpolation::Kind::None)
                        fieldJ.addCurrentToEMF<T_area>(None{});
                    else
                        fieldJ.addCurrentToEMF<T_area>(Binomial{});
                }
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
