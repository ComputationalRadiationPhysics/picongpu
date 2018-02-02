/**
* \file
* Copyright 2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
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
