/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace meta
    {
        namespace accessors
        {
            /** Get second type of the given type
             *
             * @tparam T type from which we return the second held type
             *
             * T must have defined ::second
             */
            template<typename T>
            struct Second
            {
                using type = typename T::second;
            };

        } // namespace accessors

    } // namespace meta

} // namespace  pmacc
