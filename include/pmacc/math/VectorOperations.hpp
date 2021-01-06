/* Copyright 2014-2021 Axel Huebl
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
#include "pmacc/math/Vector.hpp"

namespace pmacc
{
    namespace math
    {
        /** Map a runtime linear index to a N dimensional position
         *
         *  The size of the space to map the index to must be know at compile time
         *
         * \tparam T_Dim dimension of the position to map to
         */
        template<uint32_t T_Dim>
        struct MapToPos;

        template<>
        struct MapToPos<3>
        {
            /** Functor
             *
             *  \tparam T_ctVec math::CT::vector type like \see math::CT::Int
             *  \param math::CT::vector with spatial size to map the index to
             *  \param linearIndex linear index to be mapped
             *  \return runtime math::vector of dimension T_Dim
             */
            template<typename T_ctVec>
            DINLINE typename T_ctVec::RT_type operator()(T_ctVec, const int linearIndex)
            {
                return typename T_ctVec::RT_type(
                    (linearIndex % T_ctVec::x::value),
                    ((linearIndex % (T_ctVec::x::value * T_ctVec::y::value)) / T_ctVec::x::value),
                    (linearIndex / (T_ctVec::x::value * T_ctVec::y::value)));
            }
        };

        template<>
        struct MapToPos<2>
        {
            template<typename T_ctVec>
            DINLINE typename T_ctVec::RT_type operator()(T_ctVec, const int linearIndex)
            {
                return typename T_ctVec::RT_type((linearIndex % T_ctVec::x::value), (linearIndex / T_ctVec::x::value));
            }
        };

        template<>
        struct MapToPos<1>
        {
            template<typename T_ctVec>
            DINLINE typename T_ctVec::RT_type operator()(T_ctVec, const int linearIndex)
            {
                return typename T_ctVec::RT_type(linearIndex);
            }
        };

    } /* namespace math */
} /* namespace pmacc */
