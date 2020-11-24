/* Copyright 2013-2020 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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

#include <cstdint>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the stage of the PIC loop performing current interpolation
             *  and addition to grid values of the electromagnetic field
             */
            struct CurrentInterpolationAndAdditionToEMF
            {
                /** Compute the current created by particles and add it to the current
                 *  density
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
                    using CurrentInterpolation = fields::Solver::CurrentInterpolation;
                    CurrentInterpolation currentInterpolation;
                    using Margin = traits::GetMargin<CurrentInterpolation>;
                    DataSpace<simDim> const currentRecvLower(Margin::LowerMargin().toRT());
                    DataSpace<simDim> const currentRecvUpper(Margin::UpperMargin().toRT());

                    /* without interpolation, we do not need to access the FieldJ GUARD
                     * and can therefore overlap communication of GUARD->(ADD)BORDER & computation of CORE */
                    if(currentRecvLower == DataSpace<simDim>::create(0)
                       && currentRecvUpper == DataSpace<simDim>::create(0))
                    {
                        fieldJ.addCurrentToEMF<type::CORE>(currentInterpolation);
                        __setTransactionEvent(eRecvCurrent);
                        fieldJ.addCurrentToEMF<type::BORDER>(currentInterpolation);
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
                        fieldJ.addCurrentToEMF<type::CORE + type::BORDER>(currentInterpolation);
                    }
                    dc.releaseData(FieldJ::getName());
                }
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
