/* Copyright 2019-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/particles/algorithm/ForEach.hpp"
#include "pmacc/particles/frame_types.hpp"
#include "pmacc/functor/Interface.hpp"

#include <utility>

namespace pmacc
{
    namespace particles
    {
        namespace algorithm
        {
            /** Functor to execute an operation on all particles
             *
             * @tparam T_SpeciesOperator an operator to create the used species
             *                           with the species type as ::type
             * @tparam T_FunctorOperator an operator to create a particle functor
             *                           with the functor type as ::type
             */
            template<typename T_SpeciesOperator, typename T_FunctorOperator>
            struct CallForEach
            {
                /** Operate on the domain CORE and BORDER
                 *
                 * @param currentStep current simulation time step
                 */
                HINLINE void operator()(uint32_t const currentStep)
                {
                    using Species = typename T_SpeciesOperator::type;
                    using FrameType = typename Species::FrameType;

                    // be sure the species functor follows the pmacc functor interface
                    using UnaryFunctor = pmacc::functor::Interface<typename T_FunctorOperator::type, 1u, void>;

                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto species = dc.get<Species>(FrameType::getName(), true);

                    forEach(*species, UnaryFunctor(currentStep));

                    dc.releaseData(FrameType::getName());
                }
            };

        } // namespace algorithm
    } // namespace particles
} // namespace pmacc
