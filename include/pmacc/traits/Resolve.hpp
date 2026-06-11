/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

namespace pmacc
{
    namespace traits
    {
        /** Get resolved type
         *
         * Explicitly resolve the type of a synonym type, e.g., resolve the type of an PMacc alias.
         * A synonym type is wrapper type (class) around an other type.
         * If this trait is not defined for the given type the result is the identity of the given type.
         *
         * @tparam T_Object any object (class or typename)
         *
         * @treturn ::type
         */
        template<typename T_Object>
        struct Resolve
        {
            using type = T_Object;
        };

    } // namespace traits

} // namespace pmacc
