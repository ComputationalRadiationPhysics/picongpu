/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

namespace pmacc
{
    namespace traits
    {
        template<typename Type>
        struct GetValueType<Type*>
        {
            using ValueType = Type;
        };
    } // namespace traits
} // namespace pmacc
