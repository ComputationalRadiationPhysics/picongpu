/*
 * SPDX-FileCopyrightText: Heiko Burau
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/traits/HasFlag.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace particles
    {
        namespace traits
        {
            /** Return a new sequence of particle species carrying flag.
             *
             * @tparam T_MPLSeq sequence of particle species
             * @tparam T_Flag flag to be filtered
             */
            template<typename T_MPLSeq, typename T_Flag>
            struct FilterByFlag
            {
                template<typename T_Species>
                using HasFlag = typename ::pmacc::traits::HasFlag<typename T_Species::FrameType, T_Flag>::type;

                using type = mp_copy_if<T_MPLSeq, HasFlag>;
            };

        } // namespace traits
    } // namespace particles
} // namespace pmacc
