/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz, Alexander Grund
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
#include "pmacc/meta/conversion/SeqToMap.hpp"
#include "pmacc/meta/conversion/TypeToAliasPair.hpp"
#include "pmacc/meta/conversion/TypeToPair.hpp"
#include "pmacc/meta/errorHandlerPolicies/ReturnType.hpp"

#include <boost/mpl/at.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/type_traits/is_same.hpp>

namespace pmacc
{
    /**
     * Returns the key type from an alias
     *
     * \tparam T_MPLSeq Sequence of keys to search
     * \tparam T_Key Key or alias of a key in the sequence
     * \tparam T_KeyNotFoundPolicy Binary meta-function that is called like (T_MPLSeq, T_Key)
     *         when T_Key is not found in the sequence. Default is to return bmpl::void_
     */
    template<typename T_MPLSeq, typename T_Key, typename T_KeyNotFoundPolicy = errorHandlerPolicies::ReturnType<>>
    struct GetKeyFromAlias
    {
    private:
        typedef T_KeyNotFoundPolicy KeyNotFoundPolicy;
        /*create a map where Key is a undeclared alias and value is real type*/
        typedef typename SeqToMap<T_MPLSeq, TypeToAliasPair<bmpl::_1>>::type AliasMap;
        /*create a map where Key and value is real type*/
        typedef typename SeqToMap<T_MPLSeq, TypeToPair<bmpl::_1>>::type KeyMap;
        /*combine both maps*/
        typedef bmpl::inserter<KeyMap, bmpl::insert<bmpl::_1, bmpl::_2>> Map_inserter;
        typedef typename bmpl::copy<AliasMap, Map_inserter>::type FullMap;
        /* search for given key,
         * - we get the real type if key found
         * - else we get boost::mpl::void_
         */
        typedef typename bmpl::at<FullMap, T_Key>::type MapType;

    public:
        /* Check for KeyNotFound and calculate final type. (Uses lazy evaluation) */
        typedef typename bmpl::if_<
            boost::is_same<MapType, bmpl::void_>,
            bmpl::apply<KeyNotFoundPolicy, T_MPLSeq, T_Key>,
            bmpl::identity<MapType>>::type::type type;
    };

} // namespace pmacc
