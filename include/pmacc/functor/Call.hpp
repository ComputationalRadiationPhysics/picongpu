/* Copyright 2014-2023 Rene Widera, Sergei Bastrakov
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
            HINLINE void operator()(const uint32_t currentStep)
            {
                Functor()(currentStep);
            }
        };

    } // namespace functor
} // namespace pmacc
