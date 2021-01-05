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

#include "pmacc/math/vector/Size_t.hpp"
#include "pmacc/math/vector/Int.hpp"

#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

namespace pmacc
{
    namespace algorithm
    {
        namespace host
        {
#ifndef FOREACH_HOST_MAX_PARAMS
#    define FOREACH_HOST_MAX_PARAMS 4
#endif

#define SHIFT_CURSOR_ZONE(Z, N, _) C##N c##N##_shifted = c##N(p_zone.offset);
#define SHIFTACCESS_SHIFTEDCURSOR(Z, N, _) c##N##_shifted[cellIndex]

            namespace detail
            {
                /** Return pseudo 3D-range of the zone as math::Int<dim> */
                template<uint32_t dim>
                struct GetRange;

                template<>
                struct GetRange<3u>
                {
                    template<typename Zone>
                    const math::Int<3u> operator()(const Zone p_zone) const
                    {
                        return math::Int<3u>(p_zone.size.x(), p_zone.size.y(), p_zone.size.z());
                    }
                };
                template<>
                struct GetRange<2u>
                {
                    template<typename Zone>
                    const math::Int<3u> operator()(const Zone p_zone) const
                    {
                        return math::Int<3u>(p_zone.size.x(), p_zone.size.y(), 1);
                    }
                };
                template<>
                struct GetRange<1u>
                {
                    template<typename Zone>
                    const math::Int<3u> operator()(const Zone p_zone) const
                    {
                        return math::Int<3u>(p_zone.size.x(), 1, 1);
                    }
                };
            } // namespace detail

#define FOREACH_OPERATOR(Z, N, _)                                                                                     \
    template<typename Zone, BOOST_PP_ENUM_PARAMS(N, typename C), typename Functor, typename T_Acc>                    \
    void operator()(                                                                                                  \
        const T_Acc& acc,                                                                                             \
        const Zone& p_zone,                                                                                           \
        BOOST_PP_ENUM_BINARY_PARAMS(N, C, c),                                                                         \
        const Functor& functor)                                                                                       \
    {                                                                                                                 \
        BOOST_PP_REPEAT(N, SHIFT_CURSOR_ZONE, _)                                                                      \
                                                                                                                      \
        detail::GetRange<Zone::dim> getRange;                                                                         \
        for(int z = 0; z < getRange(p_zone).z(); z++)                                                                 \
        {                                                                                                             \
            for(int y = 0; y < getRange(p_zone).y(); y++)                                                             \
            {                                                                                                         \
                for(int x = 0; x < getRange(p_zone).x(); x++)                                                         \
                {                                                                                                     \
                    math::Int<Zone::dim> cellIndex = math::Int<3u>(x, y, z).shrink<Zone::dim>();                      \
                    functor(acc, BOOST_PP_ENUM(N, SHIFTACCESS_SHIFTEDCURSOR, _));                                     \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }

            /** Foreach algorithm (restricted to 3D)
             */
            struct Foreach
            {
                /* operator()(zone, cursor0, cursor1, ..., cursorN-1, functor or lambdaFun)
                 *
                 * \param zone Accepts currently only a zone::SphericZone object (e.g. containerObj.zone())
                 * \param cursorN cursor for the N-th data source (e.g. containerObj.origin())
                 * \param functor or lambdaFun either a functor with N arguments or a N-ary lambda function (e.g. _1 =
                 * _2)
                 *
                 * The functor or lambdaFun is called for each cell within the zone.
                 * It is called like functor(*cursor0(cellId), ..., *cursorN(cellId))
                 *
                 */
                BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_HOST_MAX_PARAMS), FOREACH_OPERATOR, _)
            };

#undef FOREACH_OPERATOR
#undef SHIFT_CURSOR_ZONE
#undef SHIFTACCESS_SHIFTEDCURSOR

        } // namespace host
    } // namespace algorithm
} // namespace pmacc
