/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/meta/conversion/ToSeq.hpp"

namespace pmacc
{
    namespace particles
    {
        namespace operations
        {
            namespace detail
            {
                /* functor for deselect attributes of an object
                 *
                 * - must define a operator()(T_Object)
                 *
                 * @tparam T_Sequence any boost mpl sequence
                 * @tparam T_Object a type were we can deselect attributes from
                 */
                template<typename T_Sequence, typename T_Object>
                struct Deselect;

            } // namespace detail

            template<typename T_Exclude, typename T_Object>
            HDINLINE decltype(auto) deselect(T_Object& object)
            {
                using DeselectSeq = ToSeq<T_Exclude>;
                using BaseType = detail::Deselect<DeselectSeq, T_Object>;

                return BaseType()(object);
            }

        } // namespace operations
    } // namespace particles
} // namespace pmacc
