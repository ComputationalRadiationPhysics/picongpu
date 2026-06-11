/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
