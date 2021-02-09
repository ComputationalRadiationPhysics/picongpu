/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
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

#include "pmacc/cuSTL/cursor/Cursor.hpp"
#include "pmacc/cuSTL/cursor/navigator/BufferNavigator.hpp"
#include "pmacc/cuSTL/cursor/navigator/CartNavigator.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace cursor
    {
        namespace tools
        {
            namespace detail
            {
                template<typename TCursor, typename Tag>
                struct SliceResult;

                template<typename TCursor>
                struct SliceResult<TCursor, tag::BufferNavigator>
                {
                    typedef Cursor<
                        typename TCursor::Accessor,
                        BufferNavigator<TCursor::Navigator::dim - 1>,
                        typename TCursor::Marker>
                        type;
                };

                template<typename TCursor>
                struct SliceResult<TCursor, tag::CartNavigator>
                {
                    typedef Cursor<
                        typename TCursor::Accessor,
                        CartNavigator<TCursor::Navigator::dim - 1>,
                        typename TCursor::Marker>
                        type;
                };

                template<typename Navi, typename NaviTag>
                struct Slice_helper;

                template<typename Navi>
                struct Slice_helper<Navi, tag::BufferNavigator>
                {
                    HDINLINE
                    BufferNavigator<Navi::dim - 1> operator()(const Navi& navi)
                    {
                        math::Size_t<Navi::dim - 2> pitch;
                        for(int i = 0; i < Navi::dim - 2; i++)
                            pitch[i] = navi.getPitch()[i];
                        return BufferNavigator<Navi::dim - 1>(pitch);
                    }
                };

                template<typename Navi>
                struct Slice_helper<Navi, tag::CartNavigator>
                {
                    HDINLINE
                    CartNavigator<Navi::dim - 1> operator()(const Navi& navi)
                    {
                        math::Int<Navi::dim - 1> factor;
                        for(uint32_t i = 0; i < Navi::dim - 1; i++)
                            factor[i] = navi.getFactor()[i];
                        return CartNavigator<Navi::dim - 1>(factor);
                    }
                };

            } // namespace detail

            /** makes a 2D cursor of a 3D vector by dropping the z-component
             */
            template<typename TCursor>
            HDINLINE typename detail::SliceResult<TCursor, typename TCursor::Navigator::tag>::type slice(
                const TCursor& cur)
            {
                detail::Slice_helper<typename TCursor::Navigator, typename TCursor::Navigator::tag> slice_helper;
                return typename detail::SliceResult<TCursor, typename TCursor::Navigator::tag>::type(
                    cur.getAccessor(),
                    slice_helper(cur.getNavigator()),
                    cur.getMarker());
            }

        } // namespace tools
    } // namespace cursor
} // namespace pmacc
