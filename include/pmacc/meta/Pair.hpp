/*
 * SPDX-FileCopyrightText: Bernhard Manfred Gruber
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

namespace pmacc::meta
{
    template<typename First, typename Second>
    struct Pair
    {
        using first = First;
        using second = Second;
    };
} // namespace pmacc::meta
