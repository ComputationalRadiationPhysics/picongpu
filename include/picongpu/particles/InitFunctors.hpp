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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/fields/Fields.def"
#include <pmacc/compileTime/conversion/TypeToPointerPair.hpp>
#include "picongpu/particles/manipulators/manipulators.def"
#include "picongpu/particles/densityProfiles/IProfile.def"
#include "picongpu/particles/Manipulate.hpp"
#include "picongpu/particles/filter/filter.def"
#include "picongpu/particles/manipulators/manipulators.def"

#include <pmacc/Environment.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/math/MapTuple.hpp>

#include <boost/mpl/if.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/accumulate.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/apply_wrap.hpp>


namespace picongpu
{

namespace particles
{

/** call a functor
 *
 * @tparam T_Functor unary lambda functor
 *                   operator() must take two params
 *                      - first: storage tuple
 *                      - second: current time step
 */
template<typename T_Functor = bmpl::_1>
struct CallFunctor
{
    using Functor = T_Functor;

    HINLINE void operator()(
        const uint32_t currentStep
    )
    {
        Functor()( currentStep );
    }
};

/** create density based on a normalized profile and a position profile
 *
 * constructor with current time step of density and position profile is called
 * after the density profile is created `fillAllGaps()` is called
 *
 * @tparam T_DensityFunctor unary lambda functor with profile description
 * @tparam T_PositionFunctor unary lambda functor with position description
 * @tparam T_SpeciesType type of the used species
 */
template<typename T_DensityFunctor, typename T_PositionFunctor, typename T_SpeciesType = bmpl::_1>
struct CreateDensity
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;


    typedef typename bmpl::apply1<T_DensityFunctor, SpeciesType>::type UserDensityFunctor;
    /* add interface for compile time interface validation*/
    typedef densityProfiles::IProfile<UserDensityFunctor> DensityFunctor;

    typedef typename bmpl::apply1<T_PositionFunctor, SpeciesType>::type UserPositionFunctor;
    /* add interface for compile time interface validation*/
    typedef manipulators::IUnary<UserPositionFunctor> PositionFunctor;

    HINLINE void operator()( const uint32_t currentStep )
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        auto speciesPtr = dc.get< SpeciesType >( FrameType::getName(), true );

        DensityFunctor densityFunctor(currentStep);
        PositionFunctor positionFunctor(currentStep);
        speciesPtr->initDensityProfile(densityFunctor, positionFunctor, currentStep);

        dc.releaseData( FrameType::getName() );
    }
};


/** derive species out of a another species
 *
 * after the species is derived `fillAllGaps()` on T_DestSpeciesType is called
 * copy all attributes from the source species except `particleId` to
 * the destination species
 *
 * @tparam T_ManipulateFunctor a pseudo-binary functor accepting two particle species:
                               destination and source, \see include/picongpu/particles/manipulators
 * @tparam T_SrcSpeciesType source species
 * @tparam T_DestSpeciesType destination species
 * @tparam T_Filter picongpu::particles::filter, particle filter type to select particles
 */
template<
    typename T_Functor,
    typename T_SrcSpeciesType,
    typename T_DestSpeciesType = bmpl::_1,
    typename T_Filter = filter::IsHandleValid
>
struct ManipulateDeriveSpecies
{
    using DestSpeciesType = T_DestSpeciesType;
    using DestFrameType = typename DestSpeciesType::FrameType;
    using SrcSpeciesType = T_SrcSpeciesType;
    using SrcFrameType = typename SrcSpeciesType::FrameType;

    using UserFunctor = typename bmpl::apply1<
        T_Functor,
        DestSpeciesType
    >::type;

    using Manipulator = manipulators::IBinary<
        UserFunctor,
        T_Filter
    >;

    HINLINE void operator()( const uint32_t currentStep )
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        auto speciesPtr = dc.get< DestSpeciesType >( DestFrameType::getName(), true );
        auto srcSpeciesPtr = dc.get< SrcSpeciesType >( SrcFrameType::getName(), true );

        Manipulator manipulator(currentStep);

        speciesPtr->deviceDeriveFrom(*srcSpeciesPtr, manipulator);

        dc.releaseData( DestFrameType::getName() );
        dc.releaseData( SrcFrameType::getName() );
    }
};


/** derive species out of a another species
 *
 * after the species is derived `fillAllGaps()` on T_DestSpeciesType is called
 * copy all attributes from the source species except `particleId` to
 * the destination species
 *
 * @tparam T_SrcSpeciesType source species
 * @tparam T_DestSpeciesType destination species
 * @tparam T_Filter picongpu::particles::filter, particle filter type to select particles
 */
template<
    typename T_SrcSpeciesType,
    typename T_DestSpeciesType = bmpl::_1,
    typename T_Filter = filter::IsHandleValid
>
struct DeriveSpecies : ManipulateDeriveSpecies<
    manipulators::generic::None,
    T_SrcSpeciesType,
    T_DestSpeciesType,
    T_Filter
>
{
};


/** call method fill all gaps of a species
 *
 * @tparam T_SpeciesType type of the species
 */
template<typename T_SpeciesType = bmpl::_1>
struct FillAllGaps
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;

    HINLINE void operator()( const uint32_t currentStep )
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        auto speciesPtr = dc.get< SpeciesType >( FrameType::getName(), true );
        speciesPtr->fillAllGaps();
        dc.releaseData( FrameType::getName() );
    }
};

} //namespace particles

} //namespace picongpu
