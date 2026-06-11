/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/meta/String.hpp>

namespace pmacc
{
    namespace traits
    {
        /** Return the compile time name
         *
         * @tparam T_Type type of the object where the name is queried
         * @return ::type name of the object as pmacc::meta::String,
         *         empty string is returned if the trait is not specified for
         *         T_Type
         */
        template<typename T_Type>
        struct GetCTName
        {
            using type = pmacc::meta::String<>;
        };

        template<typename T_Type>
        using GetCTName_t = typename GetCTName<T_Type>::type;

    } // namespace traits
} // namespace pmacc
