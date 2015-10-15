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

#include <types.h>
#include <boost/mpl/map.hpp>
#include <boost/mpl/erase_key.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/advance.hpp>
#include <boost/mpl/begin.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_shifted_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/static_assert.hpp>

namespace PMacc
{
namespace math
{

#ifndef MAPTUPLE_MAX_DIM
#define MAPTUPLE_MAX_DIM 8
#endif

#define CONSTRUCTOR(Z, N, _) \
    template<BOOST_PP_ENUM_PARAMS(N, typename Arg)> \
    HDINLINE \
    MapTuple(BOOST_PP_ENUM_BINARY_PARAMS(N, const Arg, &arg)) \
    : data(arg0), \
      base(BOOST_PP_ENUM_SHIFTED_PARAMS(N, arg)) \
    { \
        BOOST_STATIC_ASSERT(dim == N); \
    }


namespace mpl = boost::mpl;

template<typename ValueType_>
struct AlignedData
{
    typedef ValueType_ ValueType;
    PMACC_ALIGN(value, ValueType);

    HDINLINE AlignedData()
    {
    }

    HDINLINE AlignedData(const ValueType& value) : value(value)
    {
    }
};

template<typename ValueType_>
struct NativeData
{
    typedef ValueType_ ValueType;
    ValueType value;

    HDINLINE NativeData()
    {
    }

    HDINLINE NativeData(const ValueType& value) : value(value)
    {
    }
};

template<typename Map_, template<typename> class PODType = NativeData, bool ListEmpty = mpl::empty<Map_>::type::value>
class MapTuple;

template<typename Map_, template<typename> class PODType>
class MapTuple<Map_, PODType, true>
{
};

template<typename Map_, template<typename> class PODType>
class MapTuple<Map_, PODType, false>
: public MapTuple<typename mpl::erase_key<
Map_, typename mpl::deref<typename mpl::begin<Map_>::type>::type::first>::type, PODType
>
{
public:
    typedef Map_ Map;
    BOOST_STATIC_CONSTEXPR int dim = mpl::size<Map>::type::value;
private:
   
    typedef MapTuple<typename mpl::erase_key<
    Map, typename mpl::deref<typename mpl::begin<Map>::type>::type::first>::type, PODType> base;

    typedef typename mpl::deref<typename mpl::begin<Map>::type>::type::first Key;
    typedef typename mpl::deref<typename mpl::begin<Map>::type>::type::second Value;

    PODType<Value> data;
public:

    template<class> struct result;

    template<class F, class TKey>
    struct result<F(TKey)>
    {
        typedef typename mpl::at<Map, TKey>::type& type;
       // typedef typename mpl::at<F, TKey>::type& type2;
    };

    template<class F, class TKey>
    struct result<const F(TKey)>
    {

        typedef const typename mpl::at<Map, TKey>::type& type;
    };
/*
    template<class T_Map,template<typename> class T_PODType,bool T_isEnd,class TKey>
    struct result<const MapTuple<T_Map,T_PODType,T_isEnd>(TKey)>
    {
        typedef const typename mpl::at<Map, TKey>::type& type;
    };
*/
    HDINLINE MapTuple()
    {
    }

    template<typename Arg0>
    HDINLINE MapTuple(const Arg0& arg0) : data(arg0)
    {
        BOOST_STATIC_ASSERT(dim == 1);
    }

    BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(MAPTUPLE_MAX_DIM), CONSTRUCTOR, _)


    HDINLINE Value& operator[](const Key)
    {
        return this->data.value;
    }

    HDINLINE const Value& operator[](const Key) const
    {
        return this->data.value;
    }

    template<typename TKey>
    HDINLINE
    typename mpl::at<Map, TKey>::type&
    operator[](const TKey)
    {
        return base::operator[](TKey());
    }

    template<typename TKey>
    HDINLINE
    const
    typename mpl::at<Map, TKey>::type&
    operator[](const TKey) const
    {
        return base::operator[](TKey());
    }

    template<int i>
    HDINLINE
    typename mpl::deref<
    typename mpl::advance<
    typename mpl::begin<Map>::type, mpl::int_<i> >::type>::type::second&
    at()
    {
        return (*this)[
            typename mpl::deref<
            typename mpl::advance<
            typename mpl::begin<Map>::type, mpl::int_<i> >::type>::type::first()];
    }
};

#undef CONSTRUCTOR

#define PAIR_LIST(Z, N, _) mpl::pair<Key ## N, Value ## N>

#define MAKE_MAPTUPLE(Z, N, _) \
    template<BOOST_PP_ENUM_PARAMS(N, typename Key), BOOST_PP_ENUM_PARAMS(N, typename Value)> \
    HDINLINE \
    MapTuple<mpl::map<BOOST_PP_ENUM(N, PAIR_LIST, _)> > \
    make_MapTuple(BOOST_PP_ENUM_BINARY_PARAMS(N, const Value, &value)) \
    { \
        return MapTuple<mpl::map<BOOST_PP_ENUM(N, PAIR_LIST, _)> > \
            (BOOST_PP_ENUM_PARAMS(N, value)); \
    }

BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(MAPTUPLE_MAX_DIM), MAKE_MAPTUPLE, _)

#undef MAKE_MAPTUPLE
#undef PAIR_LIST

} // math
} // PMacc
