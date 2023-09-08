/* Copyright 2013-2023 Rene Widera, Benjamin Worpitz, Alexander Grund
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

#include "pmacc/meta/conversion/SeqToMap.hpp"
#include "pmacc/meta/conversion/TypeToAliasPair.hpp"
#include "pmacc/meta/conversion/TypeToPair.hpp"
#include "pmacc/meta/errorHandlerPolicies/ReturnType.hpp"
#include "pmacc/types.hpp"

#include <boost/mpl/apply.hpp>

#include <type_traits>

namespace pmacc
{
    /**
     * Returns the key type from an alias
     *
     * @tparam T_MPLSeq Sequence of keys to search
     * @tparam T_Key Key or alias of a key in the sequence
     * @tparam T_KeyNotFoundPolicy Binary meta-function that is called like (T_MPLSeq, T_Key)
     *         when T_Key is not found in the sequence. Default is to return void
     */
    template<typename T_MPLSeq, typename T_Key, typename T_KeyNotFoundPolicy = errorHandlerPolicies::ReturnType<>>
    struct GetKeyFromAlias
    {
    private:
        // FIXME(bgruber): all the boost::mp11:: qualifications inside this class work around nvcc 11.0 not finding the
        // mp_* templates

        using KeyNotFoundPolicy = T_KeyNotFoundPolicy;
        /*create a map where Key is a undeclared alias and value is real type*/
        using AliasMap = typename SeqToMap<T_MPLSeq, TypeToAliasPair<boost::mpl::_1>>::type;
        /*create a map where Key and value is real type*/
        using KeyMap = typename SeqToMap<T_MPLSeq, TypeToPair<boost::mpl::_1>>::type;
        /*combine both maps*/
        using FullMap = boost::mp11::mp_fold<AliasMap, KeyMap, boost::mp11::mp_map_insert>;
        /* search for given key,
         * - we get the real type if key found
         * - else we get boost::mpl::void_
         */
        using MapType = boost::mp11::mp_map_find<FullMap, T_Key>;

    public:
        /* Check for KeyNotFound and calculate final type. (Uses lazy evaluation) */
        using type = typename boost::mp11::mp_if<
            std::is_same<MapType, void>,
            boost::mpl::apply<KeyNotFoundPolicy, T_MPLSeq, T_Key>,
            boost::mp11::mp_defer<boost::mp11::mp_second, MapType>>::type;
    };

} // namespace pmacc
