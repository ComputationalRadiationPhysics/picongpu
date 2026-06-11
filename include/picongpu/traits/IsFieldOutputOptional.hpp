/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <type_traits>

namespace picongpu
{
    namespace traits
    {
        /** Whether a field is optional for output
         *
         * Optional fields are skipped when they are requested for output, but do not exist.
         * Doing the same for a non-optional field results in an error.
         *
         * Inherits std::true_type, std::false_type or a compatible type.
         *
         * @tparam T_Field field type
         */
        template<typename T_Field>
        struct IsFieldOutputOptional : std::false_type
        {
        };

    } // namespace traits
} // namespace picongpu
