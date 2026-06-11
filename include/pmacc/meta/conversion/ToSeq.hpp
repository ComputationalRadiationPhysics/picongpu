/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/meta/Mp11.hpp"

namespace pmacc
{
    namespace detail
    {
        template<typename T_Type>
        struct ToSeq
        {
            using type = mp_list<T_Type>;
        };

        template<typename... Ts>
        struct ToSeq<mp_list<Ts...>>
        {
            using type = mp_list<Ts...>;
        };
    } // namespace detail

    /** If T_Type is an mp_list, return it. Otherwise wrap it in an mp_list.
     */
    template<typename T_Type>
    using ToSeq = typename detail::ToSeq<T_Type>::type;
} // namespace pmacc
