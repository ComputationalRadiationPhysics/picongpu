/*
 * SPDX-FileCopyrightText: Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            /** Derived Attribute as a function of two attributes directly derived from particles
             *
             * @tparam T_BaseDerivedAttribute first parameter (derived attribute)
             * @tparam T_ModifyingDerivedAttribute second parameter (derived attribute)
             * @tparam T_ModifyingOperation functor defining the function of the two parameters
             * @tparam T_AttributeDescription class providing unit and name for the resulting attribute
             */
            template<
                typename T_BaseDerivedAttribute,
                typename T_ModifyingDerivedAttribute,
                typename T_ModifyingOperation,
                typename T_AttributeDescription>
            struct CombinedDeriveAttribute
            {
            };
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
