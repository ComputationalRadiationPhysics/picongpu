/* Copyright 2014-2019 Rene Widera
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
#include "picongpu/particles/filter/filter.def"
#include "picongpu/particles/manipulators/manipulators.def"

#include <pmacc/Environment.hpp>
#include <pmacc/particles/compileTime/FindByNameOrType.hpp>

#include <boost/mpl/apply.hpp>


namespace picongpu
{
namespace particles
{

    /** Run a user defined manipulation for each particle of a species
     *
     * Allows to manipulate attributes of existing particles in a species with
     * arbitrary unary functors ("manipulators").
     *
     * @warning Does NOT call FillAllGaps after manipulation! If the
     *          manipulation deactivates particles or creates "gaps" in any
     *          other way, FillAllGaps needs to be called for the
     *          `T_SpeciesType` manually in the next step!
     *
     * @tparam T_Manipulator unary lambda functor accepting one particle
     *                       species,
     *                       @see picongpu::particles::manipulators
     * @tparam T_SpeciesType type or name as boost::mpl::string of the used species
     * @tparam T_Filter picongpu::particles::filter, particle filter type to
     *                  select particles in `T_SpeciesType` to manipulate via
     *                  `T_DestSpeciesType`
     */
    template<
        typename T_Manipulator,
        typename T_SpeciesType = bmpl::_1,
        typename T_Filter = filter::All
    >
    struct Manipulate
    {
        using SpeciesType = pmacc::particles::compileTime::FindByNameOrType_t<
            VectorAllSpecies,
            T_SpeciesType
        >;
        using FrameType = typename SpeciesType::FrameType;

        using SpeciesFunctor = typename bmpl::apply1<
            T_Manipulator,
            SpeciesType
        >::type;

        using SpeciesFilter = typename bmpl::apply1<
            T_Filter,
            SpeciesType
        >::type;

        using FilteredManipulator = manipulators::IUnary<
            SpeciesFunctor,
            SpeciesFilter
        >;

        HINLINE void
        operator()( const uint32_t currentStep )
        {
            DataConnector &dc = Environment<>::get().DataConnector();
            auto speciesPtr = dc.get< SpeciesType >(
                FrameType::getName(),
                true
            );

            FilteredManipulator filteredManipulator( currentStep );
            speciesPtr->manipulateAllParticles(
                currentStep,
                filteredManipulator
            );

            dc.releaseData( FrameType::getName() );
        }
    };

} //namespace particles
} //namespace picongpu
