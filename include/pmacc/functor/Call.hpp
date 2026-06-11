/*
 * SPDX-FileCopyrightText: Rene Widera, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

#include <boost/mpl/placeholders.hpp>

#include <cstdint>

namespace pmacc
{
    namespace functor
    {
        /** Wrapper functor to call a functor of the given type
         *
         * @tparam T_Functor stateless unary functor type, must be default-constructible and
         *         operator() must take the current time step as the only parameter
         */
        template<typename T_Functor = boost::mpl::_1>
        struct Call
        {
            //! Functor type
            using Functor = T_Functor;

            /** Instantiate and call the functor
             *
             * @param currentStep current time iteration
             */
            HINLINE void operator()(uint32_t const currentStep)
            {
                Functor()(currentStep);
            }
        };

    } // namespace functor
} // namespace pmacc
