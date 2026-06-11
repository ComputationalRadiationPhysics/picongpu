/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"

namespace pmacc
{
    namespace algorithms
    {
        namespace precisionCast
        {
            template<typename CastToType, typename Type>
            struct TypeCast
            {
                using result = CastToType;

                constexpr result operator()(Type const& value) const
                {
                    return static_cast<result>(value);
                }
            };

            template<typename CastToType, typename Type>
            constexpr typename TypeCast<CastToType, Type>::result precisionCast(Type const& value)
            {
                return TypeCast<CastToType, Type>()(value);
            }

        } // namespace precisionCast
    } // namespace algorithms
} // namespace pmacc
