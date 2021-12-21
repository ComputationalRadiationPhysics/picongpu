/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <type_traits>
#include <utility>

namespace alpaka
{
    struct ConceptIdxBt
    {
    };
    struct ConceptIdxGb
    {
    };

    //-----------------------------------------------------------------------------
    //! The idx traits.
    namespace traits
    {
        //#############################################################################
        //! The idx type trait.
        template<typename T, typename TSfinae = void>
        struct IdxType;
    } // namespace traits

    template<typename T>
    using Idx = typename traits::IdxType<T>::type;

    namespace traits
    {
        //#############################################################################
        //! The arithmetic idx type trait specialization.
        template<typename T>
        struct IdxType<T, std::enable_if_t<std::is_arithmetic<T>::value>>
        {
            using type = std::decay_t<T>;
        };

        //#############################################################################
        //! The index get trait.
        template<typename TIdx, typename TOrigin, typename TUnit, typename TSfinae = void>
        struct GetIdx;
    } // namespace traits
} // namespace alpaka
