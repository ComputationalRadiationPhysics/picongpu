/**
 * Copyright 2013, 2015 Rene Widera, Benjamin Worpitz
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

#include "compileTime/conversion/SeqToMap.hpp"
#include "compileTime/conversion/TypeToAliasPair.hpp"
#include "compileTime/conversion/TypeToPair.hpp"
#include "static_assert.hpp"
#include "types.h"

#include <boost/mpl/at.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/type_traits/is_same.hpp>

namespace PMacc
{

template<typename T_MPLSeq,
         typename T_Key
>
struct GetKeyFromAlias
{

    /*create a map where Key is a undeclared alias and value is real type*/
    typedef typename SeqToMap<T_MPLSeq,TypeToAliasPair<bmpl::_1> >::type AliasMap;
    /*create a map where Key and value is real type*/
    typedef typename SeqToMap<T_MPLSeq,TypeToPair<bmpl::_1> >::type KeyMap;
    /*combine both maps*/
    typedef bmpl::inserter< KeyMap, bmpl::insert<bmpl::_1, bmpl::_2> > Map_inserter;
    typedef typename bmpl::copy<
        AliasMap,
        Map_inserter
        >::type FullMap;
    /* search for given key,
     * - we get the real type if key found
     * - else we get boost::mpl::void_
     */
    typedef typename bmpl::at<FullMap,T_Key>::type type;
};

template<typename T_MPLSeq,
        typename T_Key
>
struct GetKeyFromAlias_assert
{
    typedef typename GetKeyFromAlias<T_MPLSeq,T_Key>::type type;
    /*this assert fails if T_Key was not found*/
    PMACC_CASSERT_MSG_TYPE(key_not_found,T_Key,(!boost::is_same<type,bmpl::void_>::value));
};

}//namespace PMacc
