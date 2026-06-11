/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/MaxwellSolver/Substepping/Substepping.def"
#include "picongpu/param/fieldSolver.param"

namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Functor to compile-time get time step used inside the given field solver
             *
             * The default implementation uses same time step as in general PIC.
             *
             * @tparam T_FieldSolver field solver typedef
             */
            template<typename T_FieldSolver>
            struct GetTimeStep
            {
                //! Get the time step value
                HDINLINE constexpr float_X operator()()
                {
                    return sim.pic.getDt();
                }
            };

            /** Specialization of functor to compile-time get time step used inside a substepping field solver
             *
             * @tparam T_BaseSolver base field solver, follows requirements of field solvers
             * @tparam T_numSubsteps number of substeps per PIC time iteration
             */
            template<typename T_BaseSolver, uint32_t T_numSubsteps>
            struct GetTimeStep<Substepping<T_BaseSolver, T_numSubsteps>>
            {
                //! Get the time step value
                HDINLINE constexpr float_X operator()()
                {
                    return sim.pic.getDt() / static_cast<float_X>(T_numSubsteps);
                }
            };

            //! Get time step used inside the field solver
            HDINLINE constexpr float_X getTimeStep()
            {
                return GetTimeStep<Solver>{}();
            }

        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
