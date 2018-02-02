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

namespace alpaka
{
    namespace meta
    {
        //-----------------------------------------------------------------------------
        // This is based on code by Patrick Fromberg.
        // See http://stackoverflow.com/questions/9122028/how-to-create-the-cartesian-product-of-a-type-list/19611856#19611856
        namespace detail
        {
            //#############################################################################
            template<
                typename... Ts>
            struct CartesianProductImplHelper;
            //#############################################################################
            // Stop condition.
            template<
                template<typename...> class TList,
                typename... Ts>
            struct CartesianProductImplHelper<
                TList<Ts...>>
            {
                using type = TList<Ts...>;
            };
            //#############################################################################
            // Catches first empty tuple.
            template<
                template<typename...> class TList,
                typename... Ts>
            struct CartesianProductImplHelper<
                TList<TList<>>,
                Ts...>
            {
                using type = TList<>;
            };
            //#############################################################################
            // Catches any empty tuple except first.
            template<
                template<typename...> class TList,
                typename... Ts,
                typename... Rests>
            struct CartesianProductImplHelper<
                TList<Ts...>,
                TList<>,
                Rests...>
            {
                using type = TList<>;
            };
            //#############################################################################
            template<
                template<typename...> class TList,
                typename... X,
                typename H,
                typename... Rests>
            struct CartesianProductImplHelper<
                TList<X...>,
                TList<H>,
                Rests...>
            {
                using type1 = TList<Concatenate<X, TList<H>>...>;
                using type = typename CartesianProductImplHelper<type1, Rests...>::type;
            };
            //#############################################################################
            template<
                template<typename...> class TList,
                typename... X,
                template<typename...> class Head,
                typename T,
                typename... Ts,
                typename... Rests>
            struct CartesianProductImplHelper<
                TList<X...>,
                Head<T, Ts...>,
                Rests...>
            {
                using type1 = TList<Concatenate<X, TList<T>>...>;
                using type2 = typename CartesianProductImplHelper<TList<X...>, TList<Ts...>>::type;
                using type3 = Concatenate<type1, type2>;
                using type = typename CartesianProductImplHelper<type3, Rests...>::type;
            };

            //#############################################################################
            template<
                template<typename...> class TList,
                typename... Ts>
            struct CartesianProductImpl;
            //#############################################################################
            // The base case for no input returns an empty sequence.
            template<
                template<typename...> class TList>
            struct CartesianProductImpl<
                TList>
            {
                using type = TList<>;
            };
            //#############################################################################
            // R is the return type, Head<A...> is the first input list
            template<
                template<typename...> class TList,
                template<typename...> class Head,
                typename... Ts,
                typename... Tail>
            struct CartesianProductImpl<
                TList,
                Head<Ts...>,
                Tail...>
            {
                using type =
                    typename detail::CartesianProductImplHelper<
                        TList<TList<Ts>...>,
                        Tail...
                    >::type;
            };
        }

        //#############################################################################
        template<
            template<typename...> class TList,
            typename... Ts>
        using CartesianProduct = typename detail::CartesianProductImpl<TList, Ts...>::type;
    }
}
