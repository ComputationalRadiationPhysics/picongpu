/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/math/operation/traits.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace math
    {
        namespace operation
        {
            struct Sub
            {
                template<typename Dst, typename Src>
                HDINLINE constexpr void operator()(Dst& dst, Src const& src) const
                {
                    dst -= src;
                }

                template<typename Dst, typename Src, typename T_Worker>
                HDINLINE constexpr void operator()(T_Worker const&, Dst& dst, Src const& src) const
                {
                    dst -= src;
                }
            };

            namespace traits
            {
                template<>
                struct AlpakaAtomicOp<Sub>
                {
                    using type = alpaka::AtomicSub;
                };
            } // namespace traits

        } // namespace operation
    } // namespace math
} // namespace pmacc
