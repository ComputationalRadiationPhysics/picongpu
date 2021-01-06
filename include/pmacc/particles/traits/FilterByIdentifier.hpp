/* Copyright 2015-2021 Heiko Burau, Rene Widera
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

#include "pmacc/types.hpp"
#include "pmacc/traits/HasIdentifier.hpp"
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/placeholders.hpp>


namespace pmacc
{
    namespace particles
    {
        namespace traits
        {
            /** Return a new sequence of species which carry the identifier.
             *
             * @tparam T_MPLSeq sequence of particle species
             * @tparam T_Identifier identifier to be filtered
             *
             * @typedef type boost mpl forward sequence
             */
            template<typename T_MPLSeq, typename T_Identifier>
            struct FilterByIdentifier
            {
                using MPLSeq = T_MPLSeq;
                using Identifier = T_Identifier;

                template<typename T_Species>
                struct HasIdentifier
                {
                    using type =
                        typename ::pmacc::traits::HasIdentifier<typename T_Species::FrameType, Identifier>::type;
                };

                using type = typename bmpl::copy_if<MPLSeq, HasIdentifier<bmpl::_>>::type;
            };

        } // namespace traits
    } // namespace particles
} // namespace pmacc
