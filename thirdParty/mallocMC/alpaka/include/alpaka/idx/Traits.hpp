/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <utility>
#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The index specifics.
    namespace idx
    {
        struct ConceptIdxBt{};
        struct ConceptIdxGb{};

        //-----------------------------------------------------------------------------
        //! The idx traits.
        namespace traits
        {
            //#############################################################################
            //! The idx type trait.
            template<
                typename T,
                typename TSfinae = void>
            struct IdxType;
        }

        template<
            typename T>
        using Idx = typename traits::IdxType<T>::type;

        //-----------------------------------------------------------------------------
        // Trait specializations for unsigned integral types.
        namespace traits
        {
            //#############################################################################
            //! The arithmetic idx type trait specialization.
            template<
                typename T>
            struct IdxType<
                T,
                std::enable_if_t<std::is_arithmetic<T>::value>>
            {
                using type = std::decay_t<T>;
            };
        }

        //-----------------------------------------------------------------------------
        //! The index traits.
        namespace traits
        {
            //#############################################################################
            //! The index get trait.
            template<
                typename TIdx,
                typename TOrigin,
                typename TUnit,
                typename TSfinae = void>
            struct GetIdx;
        }
    }
}
