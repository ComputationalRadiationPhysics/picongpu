/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "Cursor.hpp"
#include "accessor/FunctorAccessor.hpp"
#include "navigator/CursorNavigator.hpp"
#include "lambda/make_Functor.hpp"
#include <boost/type_traits/remove_reference.hpp>

namespace PMacc
{
namespace cursor
{

/** wraps a cursor into a new cursor
 *
 * On each access of the new cursor the result of the nested cursor access
 * is filtered through a user-defined functor.
 * e.g.: new_cur = make_FunctorCursor(cur, _1[2]); // access just the z-component
 *
 * \param cursor Cursor to be wrapped
 * \param functor User functor acting as a filter. A lambda expression is allowed too.
 */
template<typename TCursor, typename Functor>
HDINLINE
Cursor<FunctorAccessor<typename lambda::result_of::make_Functor<Functor>::type,
    typename boost::remove_reference<typename TCursor::type>::type>,
    CursorNavigator, TCursor> make_FunctorCursor(const TCursor& cursor, const Functor& functor)
{
    return make_Cursor(FunctorAccessor<typename lambda::result_of::make_Functor<Functor>::type,
        typename boost::remove_reference<typename TCursor::type>::type>(lambda::make_Functor(functor)),
        CursorNavigator(), cursor);
}

} // cursor
} // PMacc

