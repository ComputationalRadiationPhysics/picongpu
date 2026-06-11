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
        template<typename T>
        struct GetValueType
        {
            using ValueType = typename T::ValueType;
        };
    } // namespace traits
} // namespace pmacc

#include "GetValueType.tpp"
