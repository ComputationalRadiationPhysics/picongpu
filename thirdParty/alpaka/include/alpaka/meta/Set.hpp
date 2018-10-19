/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <utility>

namespace alpaka
{
    namespace meta
    {
        namespace detail
        {
            //#############################################################################
            //! Empty dependent type.
            template<
                typename T>
            struct Empty
            {};

            //#############################################################################
            template<
                typename... Ts>
            struct IsParameterPackSetImpl;
            //#############################################################################
            template<>
            struct IsParameterPackSetImpl<>
            {
                static constexpr bool value = true;
            };
            //#############################################################################
            // Based on code by Roland Bock: https://gist.github.com/rbock/ad8eedde80c060132a18
            // Linearly inherits from empty<T> and checks if it has already inherited from this type.
            template<
                typename T,
                typename... Ts>
            struct IsParameterPackSetImpl<T, Ts...> :
                public IsParameterPackSetImpl<Ts...>,
                public virtual Empty<T>
            {
                using Base = IsParameterPackSetImpl<Ts...>;

                static constexpr bool value = Base::value && !std::is_base_of<Empty<T>, Base>::value;
            };
        }
        //#############################################################################
        //! Trait that tells if the parameter pack contains only unique (no equal) types.
        template<
            typename... Ts>
        using IsParameterPackSet = detail::IsParameterPackSetImpl<Ts...>;

        namespace detail
        {
            //#############################################################################
            template<
                typename TList>
            struct IsSetImpl;
            //#############################################################################
            template<
                template<typename...> class TList,
                typename... Ts>
            struct IsSetImpl<
                TList<Ts...>>
            {
                static constexpr bool value = IsParameterPackSet<Ts...>::value;
            };
        }
        //#############################################################################
        //! Trait that tells if the template contains only unique (no equal) types.
        template<
            typename TList>
        using IsSet = detail::IsSetImpl<TList>;
    }
}
