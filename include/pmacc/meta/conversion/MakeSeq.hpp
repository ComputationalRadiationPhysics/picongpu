/* Copyright 2013-2023 Rene Widera, Bernhard Manfred Gruber
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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
    /** Makes an mp_list from T_Args. If any type in T_Args is a list itself, it will be unwrapped.
     */
    template<typename... T_Args>
    using MakeSeq_t = mp_flatten<mp_list<T_Args...>>;
} // namespace pmacc
