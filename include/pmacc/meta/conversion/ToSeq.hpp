/* Copyright 2013-2023 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
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
