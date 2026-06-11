/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
