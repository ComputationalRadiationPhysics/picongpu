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
            /** Get first type of the given type
             *
             * @tparam T type from which we return the first held type
             *
             * T must have defined ::first
             */
            template<typename T>
            struct First
            {
                using type = typename T::first;
            };

        } // namespace accessors

    } // namespace meta

} // namespace  pmacc
