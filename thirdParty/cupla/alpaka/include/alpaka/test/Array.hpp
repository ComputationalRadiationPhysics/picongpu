/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/alpaka.hpp"

#include <cstddef>

namespace alpaka::test
{
    template<typename TType, size_t TSize>
    struct Array
    {
        TType m_data[TSize];

        template<typename T_Idx>
        ALPAKA_FN_HOST_ACC auto operator[](const T_Idx idx) const -> TType const&
        {
            return m_data[idx];
        }

        template<typename TIdx>
        ALPAKA_FN_HOST_ACC auto operator[](const TIdx idx) -> TType&
        {
            return m_data[idx];
        }
    };
} // namespace alpaka::test
