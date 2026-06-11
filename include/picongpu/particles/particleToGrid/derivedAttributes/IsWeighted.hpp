/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <type_traits>

namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace derivedAttributes
            {
                /** Derive attribute trait whether its value is scaled to macroparticle weight
                 *
                 * Note that it describes how the derived attribute itself is calculated, not the eventual field.
                 * Logically it roughly corresponds to openPMD macroWeighted = true and weightingPower = 1.0
                 * @see traits::MacroWeighted @see traits::WeightingPower.
                 * However, as derived attributes are a separate quantity from species we have a separate trait.
                 *
                 * Inherits std::true_type, std::false_type or a compatible type.
                 *
                 * @tparam T_DerivedAttribute derived attribute type
                 */
                template<typename T_DerivedAttribute>
                struct IsWeighted : public std::false_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
