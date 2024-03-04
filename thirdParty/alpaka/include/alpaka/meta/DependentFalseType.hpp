/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace alpaka::meta
{
    //! A false_type being dependent on a ignored template parameter.
    //! This allows to use static_assert in uninstantiated template specializations without triggering.
    template<typename T>
    struct DependentFalseType : std::false_type
    {
    };
} // namespace alpaka::meta
