/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Functor to check the Courant-Friedrichs-Lewy-Condition for the given field solver
             *
             * Performs either a compile-time check or a run-time check and throws if failed.
             *
             * @tparam T_FieldSolver field solver type
             * @tparam T_Defer technical parameter to defer evaluation;
             *                 is needed for specializations with non-template solver classes
             */
            template<typename T_FieldSolver, typename T_Defer = void>
            struct CFLChecker
            {
                /** Check the CFL condition
                 *
                 * @return upper bound on `c * dt` due to chosen cell size according to CFL condition
                 */
                float_X operator()() const;
            };

        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
