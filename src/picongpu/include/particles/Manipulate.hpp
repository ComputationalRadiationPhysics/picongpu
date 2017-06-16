/* Copyright 2014-2017 Rene Widera
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

#include "simulation_defines.hpp"
#include "particles/manipulators/manipulators.def"

#include "Environment.hpp"

#include <boost/mpl/apply.hpp>


namespace picongpu
{
namespace particles
{

    /** run a user defined functor for every particle
     *
     * - constructor with current time step is called for the functor on the host side
     * - \warning `fillAllGaps()` is not called
     *
     * @tparam T_Functor unary lambda functor
     * @tparam T_SpeciesType type of the used species
     */
    template<
        typename T_Functor,
        typename T_SpeciesType = bmpl::_1
    >
    struct Manipulate
    {
        using SpeciesType = T_SpeciesType;
        using FrameType = typename SpeciesType::FrameType;

        using UserFunctor = typename bmpl::apply1<
            T_Functor,
            SpeciesType
        >::type;
        using Functor = manipulators::IManipulator< UserFunctor >;

        HINLINE void
        operator()( const uint32_t currentStep )
        {
            DataConnector &dc = Environment<>::get().DataConnector();
            auto speciesPtr = dc.get< SpeciesType >( FrameType::getName(), true );

            Functor functor( currentStep );
            speciesPtr->manipulateAllParticles(
                currentStep,
                functor
            );

            dc.releaseData( FrameType::getName() );
        }
    };

} //namespace particles
} //namespace picongpu
