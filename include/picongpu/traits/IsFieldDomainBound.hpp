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
        /** Whether a field is geometrically bound to the domain decomposition
         *  with respect to size, guard size, and offset
         *
         * Inherits std::true_type, std::false_type or a compatible type.
         *
         * @tparam T_Field field type
         */
        template<typename T_Field>
        struct IsFieldDomainBound : std::true_type
        {
        };

    } // namespace traits
} // namespace picongpu
