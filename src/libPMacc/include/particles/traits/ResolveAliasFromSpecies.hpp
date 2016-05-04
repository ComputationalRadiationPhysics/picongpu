/**
 * Copyright 2016 Heiko Burau
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
#include "pmacc_types.hpp"
#include "particles/memory/frames/Frame.hpp"
#include "traits/GetFlagType.hpp"
#include "traits/Resolve.hpp"
#include "identifier/alias.hpp"

namespace PMacc
{
namespace particles
{
namespace traits
{

/** Resolves a custom alias in the flag list of a particle species.
 *
 * \tparam T_SpeciesType particle species
 * \tparam T_Alias alias
 */
template<typename T_SpeciesType, typename T_Alias>
struct ResolveAliasFromSpecies;

template<typename T_SpeciesType, template<typename,typename> class T_Object, typename T_AnyType>
struct ResolveAliasFromSpecies<T_SpeciesType, T_Object<T_AnyType,PMacc::pmacc_isAlias> >
{
    typedef T_SpeciesType SpeciesType;
    typedef T_Object<T_AnyType,PMacc::pmacc_isAlias> Alias;
    typedef typename SpeciesType::FrameType FrameType;

    /* The following line only fetches the alias */
    typedef typename PMacc::traits::GetFlagType<FrameType, Alias >::type FoundAlias;

    /* This now resolves the alias into the actual object type */
    typedef typename PMacc::traits::Resolve<FoundAlias>::type type;
}; // struct ResolveAlias

} // namespace traits
} // namespace particles
} // namespace PMacc
