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
        /** Get Flag of an Object
         *
         * @tparam T_Object any object (class or typename)
         * @tparam T_Key a class which is used as identifier
         *
         * @treturn ::type
         */
        template<typename T_Object, typename T_Key>
        struct GetFlagType;


    } // namespace traits

} // namespace pmacc
