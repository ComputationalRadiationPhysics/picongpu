/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/types.hpp>

namespace picongpu
{
    namespace traits
    {
        /** Describe if a particle attribute describes the quantity of a macro
         *  particle
         *
         * Depending on the implementation of an attribute, it might be sometimes
         * useful to return a quantity regarding one of the underlying real
         * particles (@see MacroWeighted).
         *
         * This trait defines how each attribute needs to be scaled with the
         * weighting (linear, quadratic, ...) to convert between "real" and "macro"
         * particle attributes.
         *   @see http://www.openPMD.org
         *   @see http://dx.doi.org/10.5281/zenodo.33624
         *   @see https://git.io/vwlWa
         *
         * @tparam T_Identifier any picongpu identifier
         * @return \p float_64 ::get() as static public method
         *
         */
        template<typename T_Identifier>
        struct WeightingPower;

    } // namespace traits
} // namespace picongpu
