/**
 * Copyright 2014-2017 Rene Widera
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

#include "pmacc_types.hpp"
#include "simulation_defines.hpp"
#include <boost/mpl/if.hpp>
#include "traits/HasFlag.hpp"
#include "traits/GetFlagType.hpp"
#include "fields/Fields.def"
#include "math/MapTuple.hpp"
#include <boost/mpl/plus.hpp>
#include <boost/mpl/accumulate.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/apply_wrap.hpp>
#include "compileTime/conversion/TypeToPointerPair.hpp"
#include "particles/manipulators/manipulators.def"
#include "particles/densityProfiles/IProfile.def"
#include "particles/startPosition/IFunctor.def"
#include "traits/Resolve.hpp"

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
    typedef T_Functor Functor;

    template<typename T_StorageTuple>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
        Functor()(tuple, currentStep);
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
    typedef T_SpeciesType SpeciesType;
    typedef typename MakeIdentifier<SpeciesType>::type SpeciesName;


    typedef typename bmpl::apply1<T_DensityFunctor, SpeciesType>::type UserDensityFunctor;
    /* add interface for compile time interface validation*/
    typedef densityProfiles::IProfile<UserDensityFunctor> DensityFunctor;

    typedef typename bmpl::apply1<T_PositionFunctor, SpeciesType>::type UserPositionFunctor;
    /* add interface for compile time interface validation*/
    typedef startPosition::IFunctor<UserPositionFunctor> PositionFunctor;

    template<typename T_StorageTuple>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
        auto speciesPtr = tuple[SpeciesName()];
        DensityFunctor densityFunctor(currentStep);
        PositionFunctor positionFunctor(currentStep);
        speciesPtr->initDensityProfile(densityFunctor, positionFunctor, currentStep);
    }
};


/** derive species out of a another species
 *
 * after the species is derived `fillAllGaps()` on T_DestSpeciesType is called
 * copy all attributes from the source species except `particleId` to
 * the destination species
 *
 * @tparam T_ManipulateFunctor a pseudo-binary functor accepting two particle species:
                               destination and source, \see src/picongpu/include/particles/manipulators
 * @tparam T_SrcSpeciesType source species
 * @tparam T_DestSpeciesType destination species
 */
template<typename T_ManipulateFunctor, typename T_SrcSpeciesType, typename T_DestSpeciesType = bmpl::_1>
struct ManipulateDeriveSpecies
{
    typedef T_DestSpeciesType DestSpeciesType;
    typedef typename MakeIdentifier<DestSpeciesType>::type DestSpeciesName;
    typedef T_SrcSpeciesType SrcSpeciesType;
    typedef typename MakeIdentifier<SrcSpeciesType>::type SrcSpeciesName;
    typedef T_ManipulateFunctor ManipulateFunctor;

    template<typename T_StorageTuple>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
        auto speciesPtr = tuple[DestSpeciesName()];
        auto srcSpeciesPtr = tuple[SrcSpeciesName()];

        ManipulateFunctor manipulateFunctor(currentStep);

        speciesPtr->deviceDeriveFrom(*srcSpeciesPtr, manipulateFunctor);
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
 */
template<typename T_SrcSpeciesType, typename T_DestSpeciesType = bmpl::_1>
struct DeriveSpecies : ManipulateDeriveSpecies<manipulators::NoneImpl, T_SrcSpeciesType, T_DestSpeciesType>
{
};


/** run a user defined functor for every particle
 *
 * - constructor with current time step is called for the functor on the host side
 * - \warning `fillAllGaps()` is not called
 *
 * @tparam T_Functor unary lambda functor
 * @tparam T_SpeciesType type of the used species
 */
template<typename T_Functor, typename T_SpeciesType = bmpl::_1>
struct Manipulate
{
    typedef T_SpeciesType SpeciesType;
    typedef typename MakeIdentifier<SpeciesType>::type SpeciesName;

    typedef typename bmpl::apply1<T_Functor, SpeciesType>::type UserFunctor;
    typedef manipulators::IManipulator<UserFunctor> Functor;

    template<typename T_StorageTuple>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
        auto speciesPtr = tuple[SpeciesName()];
        Functor functor(currentStep);
        speciesPtr->manipulateAllParticles(currentStep, functor);
    }
};


/** call method fill all gaps of a species
 *
 * @tparam T_SpeciesType type of the species
 */
template<typename T_SpeciesType = bmpl::_1>
struct FillAllGaps
{
    typedef T_SpeciesType SpeciesType;
    typedef typename MakeIdentifier<SpeciesType>::type SpeciesName;

    template<typename T_StorageTuple>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep
                            )
    {
        auto speciesPtr = tuple[SpeciesName()];
        speciesPtr->fillAllGaps();
    }
};

} //namespace particles

} //namespace picongpu
