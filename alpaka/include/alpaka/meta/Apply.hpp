/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    namespace meta
    {
        namespace detail
        {
            //#############################################################################
            template<
                typename TList,
                template<typename...> class TApplicant>
            struct ApplyImpl;
            //#############################################################################
            template<
                template<typename...> class TList,
                template<typename...> class TApplicant,
                typename... T>
            struct ApplyImpl<
                TList<T...>,
                TApplicant>
            {
                using type =
                    TApplicant<T...>;
            };
        }
        //#############################################################################
        template<
            typename TList,
            template<typename...> class TApplicant>
        using Apply = typename detail::ApplyImpl<TList, TApplicant>::type;
    }
}
