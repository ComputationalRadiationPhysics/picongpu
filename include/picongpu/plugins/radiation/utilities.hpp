/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            namespace util
            {
                // goal: to increase readability of code

                template<typename A> /// a generic square function
                HDINLINE A square(A a)
                {
                    return a * a;
                }

                template<typename A, typename R> /// a more generic square function
                HDINLINE R square(A a)
                {
                    return a * a;
                }

                template<typename A> /// a generic cube function
                HDINLINE A cube(A a)
                {
                    return a * a * a;
                }

                template<typename A, typename R> /// a more generic cube function
                HDINLINE R cube(A a)
                {
                    return a * a * a;
                }

                template<typename A, typename R = A> /// a more generic square struct
                struct Cube
                {
                    HDINLINE R operator()(A a)
                    {
                        return a * a * a;
                    }
                };

                template<typename A, typename R = A> /// a more generic square struct
                struct Square
                {
                    HDINLINE R operator()(A a) const
                    {
                        return a * a;
                    }
                };

            } // namespace util

        } // namespace radiation

    } // namespace plugins

} // namespace picongpu
