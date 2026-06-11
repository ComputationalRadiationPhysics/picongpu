/*
 * SPDX-FileCopyrightText: Axel Huebl
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
            /** Get ::type member of the given type
             *
             * @tparam T type from which we return the type held in ::type
             *
             * T must have defined ::type
             */
            template<typename T = boost::mpl::_1>
            struct Type
            {
                using type = typename T::type;
            };

        } // namespace accessors
    } // namespace meta
} // namespace pmacc
