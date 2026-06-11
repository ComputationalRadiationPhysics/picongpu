/*
 * SPDX-FileCopyrightText: Marco Garten, Heiko Burau, Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

/** \file
 *
 * Calculation utilities to be relocated together with `plugins/radiation/utilities.hpp`
 * to a more appropriate place in a more generic pmacc-y way
 */

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            namespace util
            {
                /* power 2 function */
                template<typename A>
                HDINLINE A square(A a)
                {
                    return a * a;
                }

                /* power 2 function with different result type */
                template<typename A, typename R>
                HDINLINE R square(A a)
                {
                    return a * a;
                }

                /* power 3 function */
                template<typename A>
                HDINLINE A cube(A a)
                {
                    return a * a * a;
                }

                /* power 3 function with different result type */
                template<typename A, typename R>
                HDINLINE R cube(A a)
                {
                    return a * a * a;
                }

                /* power 4 function */
                template<typename A>
                HDINLINE A quad(A a)
                {
                    A const b = a * a;
                    return b * b;
                }

                /* power 4 function with different result type */
                template<typename A, typename R>
                HDINLINE R quad(A a)
                {
                    R const b = a * a;
                    return b * b;
                }

            } // namespace util

        } // namespace ionization
    } // namespace particles
} // namespace picongpu
