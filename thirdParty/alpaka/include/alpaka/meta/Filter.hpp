/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/meta/Concatenate.hpp>

#include <type_traits>

namespace alpaka
{
    namespace meta
    {
        namespace detail
        {
            //#############################################################################
            template<
                template<typename...> class TList,
                template<typename> class TPred,
                typename... Ts>
            struct FilterImplHelper;
            //#############################################################################
            template<
                template<typename...> class TList,
                template<typename> class TPred>
            struct FilterImplHelper<
                TList,
                TPred>
            {
                using type = TList<>;
            };
            //#############################################################################
            template<
                template<typename...> class TList,
                template<typename> class TPred,
                typename T,
                typename... Ts>
            struct FilterImplHelper<
                TList,
                TPred,
                T,
                Ts...>
            {
                using type =
                    typename std::conditional<
                        TPred<T>::value,    // TODO: Remove '::value' when C++14 variable templates are supported.
                        Concatenate<TList<T>, typename FilterImplHelper<TList, TPred, Ts...>::type>,
                        typename FilterImplHelper<TList, TPred, Ts...>::type >::type;
            };

            //#############################################################################
            template<
                typename TList,
                template<typename> class TPred>
            struct FilterImpl;
            //#############################################################################
            template<
                template<typename...> class TList,
                template<typename> class TPred,
                typename... Ts>
            struct FilterImpl<
                TList<Ts...>,
                TPred>
            {
                using type =
                    typename detail::FilterImplHelper<
                        TList,
                        TPred,
                        Ts...
                    >::type;
            };
        }
        //#############################################################################
        template<
            typename TList,
            template<typename> class TPred>
        using Filter = typename detail::FilterImpl<TList, TPred>::type;
    }
}
