/* Copyright 2015-2022 Heiko Burau, Rene Widera
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

#include "pmacc/traits/HasIdentifier.hpp"
#include "pmacc/types.hpp"

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
             * @typedef type boost mp11 list sequence
             */
            template<typename T_MPLSeq, typename T_Identifier>
            struct FilterByIdentifier
            {
                template<typename T_Species>
                using HasIdentifier =
                    typename ::pmacc::traits::HasIdentifier<typename T_Species::FrameType, T_Identifier>::type;

                using type = mp_copy_if<T_MPLSeq, HasIdentifier>;
            };

        } // namespace traits
    } // namespace particles
} // namespace pmacc
