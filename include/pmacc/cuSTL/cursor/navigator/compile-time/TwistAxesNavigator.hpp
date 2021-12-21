/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include "pmacc/math/Vector.hpp"

namespace pmacc
{
    namespace cursor
    {
        namespace CT
        {
            template<typename Axes, int dim = Axes::dim>
            struct TwistAxesNavigator;

            template<typename Axes>
            struct TwistAxesNavigator<Axes, 2>
            {
                static constexpr int dim = 2;

                template<typename TCursor>
                HDINLINE TCursor operator()(const TCursor& cursor, const math::Int<2>& jump) const
                {
                    math::Int<2> twistedJump;
                    twistedJump[Axes::x::value] = jump.x();
                    twistedJump[Axes::y::value] = jump.y();
                    return cursor(twistedJump);
                }
            };

            template<typename Axes>
            struct TwistAxesNavigator<Axes, 3>
            {
                static constexpr int dim = 3;

                template<typename TCursor>
                HDINLINE TCursor operator()(const TCursor& cursor, const math::Int<3>& jump) const
                {
                    math::Int<3> twistedJump;
                    twistedJump[Axes::x::value] = jump.x();
                    twistedJump[Axes::y::value] = jump.y();
                    twistedJump[Axes::z::value] = jump.z();
                    return cursor(twistedJump);
                }
            };

        } // namespace CT
    } // namespace cursor
} // namespace pmacc
