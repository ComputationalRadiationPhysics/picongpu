/* Copyright 2015-2021 Heiko Burau
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

namespace pmacc
{
    namespace cursor
    {
        /** Navigator which combines jump and marker of any type by using the +operator.
         */
        struct PlusNavigator
        {
            template<typename Marker, typename Jump>
            HDINLINE Marker operator()(const Marker& marker, const Jump& jump) const
            {
                return marker + jump;
            }
        };

    } // namespace cursor
} // namespace pmacc
