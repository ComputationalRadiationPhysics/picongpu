/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace particles
    {
        namespace operations
        {
            namespace detail
            {
                template<typename T_Dest, typename T_Src>
                struct Assign;

            } // namespace detail

            template<typename T_Dest, typename T_Src>
            HDINLINE void assign(T_Dest& dest, T_Src const& src)
            {
                detail::Assign<T_Dest, T_Src>()(dest, src);
            }

        } // namespace operations
    } // namespace particles
} // namespace pmacc
