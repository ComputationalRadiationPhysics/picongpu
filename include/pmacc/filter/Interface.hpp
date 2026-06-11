/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/functor/Interface.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace filter
    {
        /** Interface for a filter
         *
         * A filter is a functor which is evaluated to true or false depending
         * on the input parameters.
         * A filter can be used to decide e.g. if a particle is located in a user
         * defined area or if an attribute is above a threshold.
         *
         * @tparam T_UserFunctor pmacc::functor::Interface, type of the functor (filter rule)
         * @tparam T_numArguments number of arguments which must be supported by T_UserFunctor
         */
        template<typename T_UserFunctor, uint32_t T_numArguments>
        using Interface = pmacc::functor::Interface<T_UserFunctor, T_numArguments, bool>;

    } // namespace filter
} // namespace pmacc
