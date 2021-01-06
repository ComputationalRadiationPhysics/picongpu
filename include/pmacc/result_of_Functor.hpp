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

#include <boost/mpl/void.hpp>

namespace mpl = boost::mpl;

namespace pmacc
{
    namespace result_of
    {
        template<
            typename _Functor,
            typename Arg0 = mpl::void_,
            typename Arg1 = mpl::void_,
            typename Arg2 = mpl::void_,
            typename Arg3 = mpl::void_,
            typename Arg4 = mpl::void_,
            typename Arg5 = mpl::void_,
            typename Arg6 = mpl::void_,
            typename Arg7 = mpl::void_,
            typename Arg8 = mpl::void_,
            typename Arg9 = mpl::void_,
            typename Arg10 = mpl::void_,
            typename Arg11 = mpl::void_,
            typename Arg12 = mpl::void_,
            typename dummy = mpl::void_>
        struct Functor
        {
            typedef typename _Functor::result_type type;
        };

    } // namespace result_of
} // namespace pmacc
