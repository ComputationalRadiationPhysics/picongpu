/*
 * SPDX-FileCopyrightText: Marco Garten, Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/meta/accessors/Type.hpp>
#include <pmacc/meta/conversion/OperateOnSeq.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

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
                using FoundIonizersAlias = typename pmacc::traits::GetFlagType<FrameType, ionizers<>>::type;

                // this now resolves the alias into the actual object type, a list of ionizers
                using FoundIonizerList = typename pmacc::traits::Resolve<FoundIonizersAlias>::type;

                using type = typename pmacc::OperateOnSeq<
                    FoundIonizerList,
                    boost::mpl::apply1<boost::mpl::_1, SpeciesType>,
                    pmacc::meta::accessors::Type<>>::type;
            };

        } // namespace traits
    } // namespace particles
} // namespace picongpu
