/**
 * Copyright 2013 Heiko Burau, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
 
#ifndef ALGORITHM_HOST_FOREACH_HPP
#define ALGORITHM_HOST_FOREACH_HPP

#include "math/vector/Size_t.hpp"
#include "math/vector/Int.hpp"
#include "lambda/make_Functor.hpp"

#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

namespace PMacc
{
namespace algorithm
{
namespace host
{
    
#ifndef FOREACH_HOST_MAX_PARAMS
#define FOREACH_HOST_MAX_PARAMS 4
#endif

#define SHIFT_CURSOR_ZONE(Z, N, _) C ## N c ## N ## _shifted = c ## N ((math::Int<Zone::dim>) _zone.offset);
#define SHIFTACCESS_SHIFTEDCURSOR(Z, N, _) c ## N ## _shifted [cellIndex]

namespace detail
{
    /** Return cellIndex as math::Int<dim> */
    template<uint32_t dim>
    struct GetIndex;

    template<>
    struct GetIndex<3u>
    {
        const math::Int<3u> operator()(const int x, const int y, const int z) const
        {
            return math::Int<3u>(x, y, z);
        }
    };
    template<>
    struct GetIndex<2u>
    {
        const math::Int<2u> operator()(const int x, const int y, const int) const
        {
            return math::Int<2u>(x, y);
        }
    };
    template<>
    struct GetIndex<1u>
    {
        const math::Int<1u> operator()(const int x, const int, const int) const
        {
            return math::Int<1u>(x);
        }
    };

    /** Return max of the zone as math::Int<dim> */
    template< uint32_t dim >
    struct GetMax;

    template<>
    struct GetMax<3u>
    {
        template<typename Zone>
        const math::Int<3u> operator()(const Zone _zone) const
        {
            return math::Int<3u>(_zone.size.x(), _zone.size.y(), _zone.size.z());
        }
    };
    template<>
    struct GetMax<2u>
    {
        template<typename Zone>
        const math::Int<3u> operator()(const Zone _zone) const
        {
            return math::Int<3u>(_zone.size.x(), _zone.size.y(), 1);
        }
    };
    template<>
    struct GetMax<1u>
    {
        template<typename Zone>
        const math::Int<3u> operator()(const Zone _zone) const
        {
            return math::Int<3u>(_zone.size.x(), 1, 1);
        }
    };
} // namespace detail

#define FOREACH_OPERATOR(Z, N, _)                                              \
    template<typename Zone, BOOST_PP_ENUM_PARAMS(N, typename C), typename Functor> \
    void operator()(const Zone& _zone, BOOST_PP_ENUM_BINARY_PARAMS(N, C, c), const Functor& functor) \
    {                                                                          \
        BOOST_PP_REPEAT(N, SHIFT_CURSOR_ZONE, _)                               \
                                                                               \
        typename lambda::result_of::make_Functor<Functor>::type fun            \
            = lambda::make_Functor(functor);                                   \
        detail::GetIndex<Zone::dim> getIndex;                                  \
        detail::GetMax<Zone::dim> getMax;                                      \
        for(int z = 0; z < getMax(_zone).z(); z++)                             \
        {                                                                      \
            for(int y = 0; y < getMax(_zone).y(); y++)                         \
            {                                                                  \
                for(int x = 0; x < getMax(_zone).x(); x++)                     \
                {                                                              \
                    math::Int<Zone::dim> cellIndex = getIndex(x, y, z);        \
                    fun(BOOST_PP_ENUM(N, SHIFTACCESS_SHIFTEDCURSOR, _));       \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }
    
struct Foreach
{
    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(FOREACH_HOST_MAX_PARAMS), FOREACH_OPERATOR, _)
};

#undef FOREACH_OPERATOR
#undef SHIFT_CURSOR_ZONE
#undef SHIFTACCESS_SHIFTEDCURSOR
    
} // host
} // algorithm
} // PMacc

#endif // ALGORITHM_HOST_FOREACH_HPP
