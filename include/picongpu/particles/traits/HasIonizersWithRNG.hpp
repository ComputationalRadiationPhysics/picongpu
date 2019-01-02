/* Copyright 2017-2019 Axel Huebl, Marco Garten
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

#include <pmacc/compileTime/conversion/OperateOnSeq.hpp>
#include <pmacc/compileTime/conversion/MakeSeqFromNestedSeq.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include "picongpu/traits/UsesRNG.hpp"
#include "picongpu/particles/traits/GetIonizerList.hpp"

#include <boost/type_traits/integral_constant.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/placeholders.hpp>


namespace picongpu
{
namespace particles
{
namespace traits
{
    /** Check Ionizers for RNG Need
     *
     * Checks all species for ionizers and within those if a random number generator is needed.
     * Returns a true-valued boost integral constant in ::type if a RNG is needed.
     *
     * @tparam T_VectorSpecies sequence of (ion) species to check ionizers for
     */
    template< typename T_VectorSpecies >
    struct HasIonizersWithRNG
    {
        using VectorSpecies = T_VectorSpecies;

        // make a list of all species that can be ionized
        using VectorSpeciesWithIonizer = typename pmacc::particles::traits::FilterByFlag<
            VectorSpecies,
            ionizers<>
        >::type;

        // make a list of all ionizers that will be used by all species
        using AllUsedIonizers = typename pmacc::MakeSeqFromNestedSeq<
            typename pmacc::OperateOnSeq<
                VectorSpeciesWithIonizer,
                GetIonizerList< bmpl::_1 >
            >::type
        >::type;

        /* make a list of `boost::true_type`s and `boost::false_type`s for species that use or do
         * not use the RNG during ionization
         */
        using AllIonizersUsingRNG = typename pmacc::OperateOnSeq<
            AllUsedIonizers,
            picongpu::traits::UsesRNG< bmpl::_1 >
        >::type;

        // check if at least one RNG is needed
        using type = typename boost::mpl::contains<
            AllIonizersUsingRNG,
            boost::true_type
        >::type;
    };

} // namespace traits
} // namespace particles
} // namespace picongpu
