/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
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

    //! The idx trait.
    namespace trait
    {
        //! The idx type trait.
        template<typename T, typename TSfinae = void>
        struct IdxType;
    } // namespace trait

    template<typename T>
    using Idx = typename trait::IdxType<T>::type;

    namespace trait
    {
        //! The arithmetic idx type trait specialization.
        template<typename T>
        struct IdxType<T, std::enable_if_t<std::is_arithmetic_v<T>>>
        {
            using type = std::decay_t<T>;
        };

        //! The index get trait.
        template<typename TIdx, typename TOrigin, typename TUnit, typename TSfinae = void>
        struct GetIdx;
    } // namespace trait
} // namespace alpaka
