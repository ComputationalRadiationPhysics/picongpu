/* Copyright 2014-2023 Rene Widera, Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
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
