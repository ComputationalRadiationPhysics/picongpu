/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
