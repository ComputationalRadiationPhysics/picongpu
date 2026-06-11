/*
 * SPDX-FileCopyrightText: Rene Widera, Bernhard Manfred Gruber
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
