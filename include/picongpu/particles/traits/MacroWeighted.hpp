/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

// include "simulation_defines.hpp"
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
         * particles (false: "this attribute is not weighted accordingly for the
         * whole ensemble of particles in its macro particle) or just handle the
         * whole macro particle at once
         * (true: "this attribute is already weighted").
         *
         * This trait defines for each attribute if it needs to be scaled with the
         * weighting. *How* the scaling with weighting is applied can be seen in
         * @see WeightingPower
         *   @see http://www.openPMD.org
         *   @see http://dx.doi.org/10.5281/zenodo.33624
         *   @see https://git.io/vwlWa
         *
         * @tparam T_Identifier any picongpu identifier
         * @return \p bool ::get() as static public method
         *
         */
        template<typename T_Identifier>
        struct MacroWeighted;

    } // namespace traits
} // namespace picongpu
