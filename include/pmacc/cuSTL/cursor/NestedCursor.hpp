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

#include "accessor/MarkerAccessor.hpp"
#include "navigator/CursorNavigator.hpp"
#include "Cursor.hpp"

namespace pmacc
{
    namespace cursor
    {
        /** wraps a cursor into a new cursor in a way that accessing on the new cursor
         * means getting the nested cursor and jumping means jumping on the nested cursor.
         * \param cursor Cursor to be wrapped
         * \return A new cursor which wraps the given cursor
         */
        template<typename TCursor>
        HDINLINE Cursor<MarkerAccessor<TCursor>, CursorNavigator, TCursor> make_NestedCursor(const TCursor& cursor)
        {
            return Cursor<MarkerAccessor<TCursor>, CursorNavigator, TCursor>(
                MarkerAccessor<TCursor>(),
                CursorNavigator(),
                cursor);
        }

    } // namespace cursor
} // namespace pmacc
