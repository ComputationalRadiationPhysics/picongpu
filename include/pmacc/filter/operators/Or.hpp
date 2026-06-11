/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"

namespace pmacc
{
    namespace filter
    {
        namespace operators
        {
            //! combine all arguments by OR `||`
            struct Or
            {
                /** return a
                 *
                 * @param a a boolean value
                 * @return the input argument
                 */
                template<typename T_Arg>
                HDINLINE bool operator()(T_Arg const a) const
                {
                    return a;
                }

                /** get OR combined result
                 *
                 * @param args arguments to combine
                 * @return OR combination of all arguments
                 */
                template<typename T_Arg1, typename... T_Args>
                HDINLINE bool operator()(T_Arg1 const a, T_Args const... args) const
                {
                    return a || Or{}(args...);
                }
            };

        } // namespace operators
    } // namespace filter
} // namespace pmacc
