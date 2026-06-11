/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/types.hpp"

#include <boost/mpl/placeholders.hpp>

namespace pmacc
{
    namespace meta
    {
        namespace accessors
        {
            /** Get the type of a given type without changes
             *
             * @tparam T in type
             *
             */
            template<typename T = boost::mpl::_1>
            struct Identity
            {
                using type = T;
            };

        } // namespace accessors

    } // namespace meta

} // namespace  pmacc
