/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/meta/Pair.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** create pmacc::meta::Pair
     *
     * @tparam T_Type any type
     * @resturn ::type pmacc::meta::Pair where first and second is set to T_Type
     */
    template<typename T_Type>
    struct TypeToPair
    {
        using type = pmacc::meta::Pair<T_Type, T_Type>;
    };


} // namespace pmacc
