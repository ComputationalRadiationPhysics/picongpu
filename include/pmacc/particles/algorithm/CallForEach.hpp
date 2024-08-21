/* Copyright 2019-2023 Rene Widera, Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/functor/Interface.hpp"
#include "pmacc/particles/algorithm/ForEach.hpp"
#include "pmacc/particles/frame_types.hpp"

#include <cstdint>
#include <utility>

namespace pmacc
{
    namespace particles
    {
        namespace algorithm
        {
            /** Functor to execute an operation on all particles of a species in an area
             *
             * There are two versions of operator() that differ in how the area is defined.
             * One operates on the area given by the template parameter.
             * Another one takes a user-provided mapper factory that defines the area via the produced mapper.
             *
             * @tparam T_SpeciesOperator an operator to create the used species
             *                           with the species type as ::type
             * @tparam T_FunctorOperator an operator to create a particle functor
             *                           with the functor type as ::type
             * @tparam T_area area to process particles in
             */
            template<typename T_SpeciesOperator, typename T_FunctorOperator, uint32_t T_area = CORE + BORDER>
            struct CallForEach
            {
                /** Operate on T_area
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
                    auto idProvider = dc.get<IdProvider>("globalId");
                    auto species = dc.get<Species>(FrameType::getName());
                    forEach<T_area>(*species, UnaryFunctor(currentStep, idProvider->getDeviceGenerator()));
                }

                /** Operate on the area defined by mapper
                 *
                 * @tparam T_AreaMapperFactory factory type to construct an area mapper that defines the area to
                 * process, adheres to the AreaMapperFactory concept
                 *
                 * @param currentStep current simulation time step
                 * @param areaMapperFactory factory to construct an area mapper,
                 *                          the area is defined by the constructed mapper object
                 */
                template<typename T_AreaMapperFactory>
                HINLINE void operator()(uint32_t const currentStep, T_AreaMapperFactory const& areaMapperFactory)
                {
                    using Species = typename T_SpeciesOperator::type;
                    using FrameType = typename Species::FrameType;
                    // be sure the species functor follows the pmacc functor interface
                    using UnaryFunctor = pmacc::functor::Interface<typename T_FunctorOperator::type, 1u, void>;

                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto idProvider = dc.get<IdProvider>("globalId");
                    auto species = dc.get<Species>(FrameType::getName());
                    forEach(*species, UnaryFunctor(currentStep, idProvider->getDeviceGenerator()), areaMapperFactory);
                }
            };

        } // namespace algorithm
    } // namespace particles
} // namespace pmacc
