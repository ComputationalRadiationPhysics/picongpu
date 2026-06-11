/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "Mp11.hpp"

namespace pmacc
{
    /// Carthesian product of the given lists.
    /// Lists = [1,2],[1],[4,3]
    /// Result: [(1,1,4),(1,1,3),(2,1,4),(2,1,3)]
    template<typename... Lists>
    using AllCombinations = mp_product<mp_list, Lists...>;
} // namespace pmacc
