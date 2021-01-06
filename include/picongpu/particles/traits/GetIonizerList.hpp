/* Copyright 2014-2021 Marco Garten, Axel Huebl
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
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/meta/accessors/Type.hpp>
#include <pmacc/meta/conversion/OperateOnSeq.hpp>

#include <boost/mpl/apply.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace traits
        {
            /** Returns a sequence with ionizers for a species
             *
             * Several ionization methods can be assigned to a species which are called
             * consecutively (in the same order as the user inputs them) within a single
             * time step.
             *
             * @tparam T_SpeciesType ion species
             */
            template<typename T_SpeciesType>
            struct GetIonizerList
            {
                using SpeciesType = T_SpeciesType;
                using FrameType = typename SpeciesType::FrameType;

                // the following line only fetches the alias
                using FoundIonizersAlias = typename GetFlagType<FrameType, ionizers<>>::type;

                // this now resolves the alias into the actual object type, a list of ionizers
                using FoundIonizerList = typename pmacc::traits::Resolve<FoundIonizersAlias>::type;

                using type = typename pmacc::OperateOnSeq<
                    FoundIonizerList,
                    bmpl::apply1<bmpl::_1, SpeciesType>,
                    pmacc::meta::accessors::Type<>>::type;
            };

        } // namespace traits
    } // namespace particles
} // namespace picongpu
