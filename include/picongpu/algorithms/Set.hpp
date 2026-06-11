/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/types.hpp>

namespace picongpu
{
    using namespace pmacc;

    template<typename Type_>
    struct Set
    {
        HDINLINE Set(Type_ defaultValue) : value(defaultValue)
        {
        }

        template<typename Dst, typename T_Worker>
        HDINLINE void operator()(T_Worker const&, Dst& dst) const
        {
            dst = value;
        }

    private:
        PMACC_ALIGN(value, Type_ const);
    };
} // namespace picongpu
