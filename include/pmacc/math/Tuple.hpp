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

#include "pmacc/types.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/minus.hpp>
#include <boost/mpl/integral_c.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_shifted_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/static_assert.hpp>

namespace pmacc
{
    namespace math
    {
#ifndef TUPLE_MAX_DIM
#    define TUPLE_MAX_DIM 8
#endif

#define CONSTRUCTOR(Z, N, _)                                                                                          \
    template<BOOST_PP_ENUM_PARAMS(N, typename Arg)>                                                                   \
    HDINLINE Tuple(BOOST_PP_ENUM_BINARY_PARAMS(N, const Arg, &arg))                                                   \
        : value(arg0)                                                                                                 \
        , base(BOOST_PP_ENUM_SHIFTED_PARAMS(N, arg))                                                                  \
    {                                                                                                                 \
        BOOST_STATIC_ASSERT(dim == N);                                                                                \
    }

        namespace mpl = boost::mpl;

        template<typename TypeList, bool ListEmpty = mpl::empty<TypeList>::type::value>
        class Tuple;

        template<typename TypeList>
        class Tuple<TypeList, true>
        {
        };

        template<typename TypeList>
        class Tuple<TypeList, false> : public Tuple<typename mpl::pop_front<TypeList>::type>
        {
        public:
            static constexpr int dim = mpl::size<TypeList>::type::value;
            typedef TypeList TypeList_;

        private:
            typedef Tuple<typename mpl::pop_front<TypeList>::type> base;

            typedef typename mpl::front<TypeList>::type Value;
            typedef typename boost::remove_reference<Value>::type pureValue;

            Value value;

        public:
            HDINLINE Tuple()
            {
            }

            HDINLINE Tuple(Value arg0) : value(arg0)
            {
                BOOST_STATIC_ASSERT(dim == 1);
            }

            BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(TUPLE_MAX_DIM), CONSTRUCTOR, _)

            template<int i>
            HDINLINE typename mpl::at_c<TypeList, i>::type& at_c()
            {
                return this->at(mpl::int_<i>());
            }
            template<int i>
            HDINLINE const typename mpl::at_c<TypeList, i>::type& at_c() const
            {
                return this->at(mpl::int_<i>());
            }

            HDINLINE Value& at(mpl::int_<0>)
            {
                return value;
            }
            HDINLINE Value& at(mpl::integral_c<int, 0>)
            {
                return value;
            }

            HDINLINE const Value& at(mpl::int_<0>) const
            {
                return value;
            }
            HDINLINE const Value& at(mpl::integral_c<int, 0>) const
            {
                return value;
            }

            template<typename Idx>
            HDINLINE typename mpl::at<TypeList, Idx>::type& at(Idx)
            {
                return base::at(typename mpl::minus<Idx, mpl::int_<1>>::type());
            }

            template<typename Idx>
            HDINLINE const typename mpl::at<TypeList, Idx>::type& at(Idx) const
            {
                return base::at(typename mpl::minus<Idx, mpl::int_<1>>::type());
            }
        };

#undef CONSTRUCTOR

#define MAKE_TUPLE(Z, N, _)                                                                                           \
    template<BOOST_PP_ENUM_PARAMS(N, typename Value)>                                                                 \
    HDINLINE Tuple<mpl::vector<BOOST_PP_ENUM_PARAMS(N, Value)>> make_Tuple(                                           \
        BOOST_PP_ENUM_BINARY_PARAMS(N, Value, value))                                                                 \
    {                                                                                                                 \
        return Tuple<mpl::vector<BOOST_PP_ENUM_PARAMS(N, Value)>>(BOOST_PP_ENUM_PARAMS(N, value));                    \
    }

        BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(TUPLE_MAX_DIM), MAKE_TUPLE, _)

#undef MAKE_TUPLE

        namespace result_of
        {
            template<typename TTuple, int i>
            struct at_c
            {
                typedef typename mpl::at_c<typename TTuple::TypeList_, i>::type type;
            };
        } // namespace result_of

    } // namespace math
} // namespace pmacc
